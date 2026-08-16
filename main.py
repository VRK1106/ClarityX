import math
from os import path as osp

import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class HATModel(SRModel):
    """
    HAT-based super-resolution model wrapper.

    Provides:
        - Window-size compatible input padding
        - Standard full-image inference
        - Memory-efficient tiled inference
        - Automatic padding removal
        - Validation and metric calculation
        - Optional output image saving
    """

    # ------------------------------------------------------------------
    # Model inference helper
    # ------------------------------------------------------------------

    def _infer(self, input_tensor):
        """
        Run inference using the EMA generator when available.

        Args:
            input_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Super-resolved output.
        """
        model = getattr(self, "net_g_ema", None)

        if model is None:
            model = self.net_g

        model.eval()

        with torch.no_grad():
            result = model(input_tensor)

        return result

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def pre_process(self):
        """
        Pad the input image so that its spatial dimensions are
        divisible by the model's attention window size.
        """

        network_config = self.opt.get("network_g", {})
        window_size = network_config.get("window_size", 1)

        self.scale = self.opt.get("scale", 1)

        _, _, height, width = self.lq.shape

        self.mod_pad_h = (
            (window_size - height % window_size) % window_size
        )

        self.mod_pad_w = (
            (window_size - width % window_size) % window_size
        )

        if self.mod_pad_h == 0 and self.mod_pad_w == 0:
            self.img = self.lq
            return

        # Reflection padding avoids introducing artificial black borders.
        self.img = F.pad(
            self.lq,
            (
                0,
                self.mod_pad_w,
                0,
                self.mod_pad_h,
            ),
            mode="reflect",
        )

    # ------------------------------------------------------------------
    # Standard inference
    # ------------------------------------------------------------------

    def process(self):
        """
        Perform inference on the complete padded image.
        """

        self.output = self._infer(self.img)

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def tile_process(self):
        """
        Process large images using overlapping tiles.

        Tile padding provides contextual information around tile borders,
        while only the valid center region of each tile is copied into
        the final output.
        """

        batch_size, channels, height, width = self.img.shape

        tile_config = self.opt.get("tile", {})

        tile_size = tile_config.get("tile_size", 400)
        tile_pad = tile_config.get("tile_pad", 10)

        scale = self.scale

        output_height = height * scale
        output_width = width * scale

        self.output = self.img.new_zeros(
            (
                batch_size,
                channels,
                output_height,
                output_width,
            )
        )

        tiles_horizontal = math.ceil(width / tile_size)
        tiles_vertical = math.ceil(height / tile_size)

        total_tiles = tiles_horizontal * tiles_vertical
        current_tile = 0

        for tile_y in range(tiles_vertical):

            for tile_x in range(tiles_horizontal):

                current_tile += 1

                # ------------------------------------------------------
                # Original tile coordinates
                # ------------------------------------------------------

                start_x = tile_x * tile_size
                start_y = tile_y * tile_size

                end_x = min(start_x + tile_size, width)
                end_y = min(start_y + tile_size, height)

                # ------------------------------------------------------
                # Expand tile boundaries using contextual padding
                # ------------------------------------------------------

                padded_start_x = max(start_x - tile_pad, 0)
                padded_start_y = max(start_y - tile_pad, 0)

                padded_end_x = min(end_x + tile_pad, width)
                padded_end_y = min(end_y + tile_pad, height)

                tile_input = self.img[
                    :,
                    :,
                    padded_start_y:padded_end_y,
                    padded_start_x:padded_end_x,
                ]

                # ------------------------------------------------------
                # Run the HAT model on the tile
                # ------------------------------------------------------

                try:
                    tile_output = self._infer(tile_input)

                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        torch.cuda.empty_cache()

                        raise RuntimeError(
                            "GPU memory was insufficient while processing "
                            f"tile {current_tile}/{total_tiles}. "
                            "Try reducing tile_size."
                        ) from error

                    raise

                # ------------------------------------------------------
                # Determine valid region inside the processed tile
                # ------------------------------------------------------

                valid_width = end_x - start_x
                valid_height = end_y - start_y

                crop_start_x = (start_x - padded_start_x) * scale
                crop_start_y = (start_y - padded_start_y) * scale

                crop_end_x = crop_start_x + valid_width * scale
                crop_end_y = crop_start_y + valid_height * scale

                valid_tile = tile_output[
                    :,
                    :,
                    crop_start_y:crop_end_y,
                    crop_start_x:crop_end_x,
                ]

                # ------------------------------------------------------
                # Destination coordinates in final image
                # ------------------------------------------------------

                output_start_x = start_x * scale
                output_start_y = start_y * scale

                output_end_x = end_x * scale
                output_end_y = end_y * scale

                self.output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = valid_tile

                print(
                    f"\rProcessing tile "
                    f"{current_tile}/{total_tiles}",
                    end="",
                )

        print()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self):
        """
        Remove the padding that was introduced during pre-processing.
        """

        if self.mod_pad_h == 0 and self.mod_pad_w == 0:
            return

        _, _, output_height, output_width = self.output.shape

        crop_height = self.mod_pad_h * self.scale
        crop_width = self.mod_pad_w * self.scale

        final_height = output_height - crop_height
        final_width = output_width - crop_width

        self.output = self.output[
            :,
            :,
            :final_height,
            :final_width,
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def nondist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img,
    ):
        """
        Run validation without distributed processing.

        Supports:
            - Full-image inference
            - Tile-based inference
            - Image saving
            - Metric calculation
            - Progress display
        """

        dataset_name = dataloader.dataset.opt["name"]

        validation_config = self.opt.get("val", {})

        metrics_config = validation_config.get("metrics")
        calculate_metrics = metrics_config is not None

        show_progress = validation_config.get("pbar", False)

        # --------------------------------------------------------------
        # Metric initialization
        # --------------------------------------------------------------

        if calculate_metrics:

            if not hasattr(self, "metric_results"):
                self.metric_results = {
                    metric_name: 0
                    for metric_name in metrics_config.keys()
                }

            self._initialize_best_metric_results(
                dataset_name
            )

            self.metric_results = {
                metric_name: 0
                for metric_name in self.metric_results.keys()
            }

        # --------------------------------------------------------------
        # Progress bar
        # --------------------------------------------------------------

        progress = None

        if show_progress:
            progress = tqdm(
                total=len(dataloader),
                unit="image",
                desc="Validation",
            )

        # --------------------------------------------------------------
        # Validation loop
        # --------------------------------------------------------------

        for index, validation_data in enumerate(dataloader):

            image_path = validation_data["lq_path"][0]

            image_name = osp.splitext(
                osp.basename(image_path)
            )[0]

            # ----------------------------------------------------------
            # Load input
            # ----------------------------------------------------------

            self.feed_data(validation_data)

            # ----------------------------------------------------------
            # Prepare image
            # ----------------------------------------------------------

            self.pre_process()

            # ----------------------------------------------------------
            # Select inference strategy
            # ----------------------------------------------------------

            if "tile" in self.opt:
                self.tile_process()
            else:
                self.process()

            # ----------------------------------------------------------
            # Restore original image dimensions
            # ----------------------------------------------------------

            self.post_process()

            # ----------------------------------------------------------
            # Convert tensors to images
            # ----------------------------------------------------------

            visuals = self.get_current_visuals()

            sr_image = tensor2img(
                [visuals["result"]]
            )

            metric_data = {
                "img": sr_image
            }

            # ----------------------------------------------------------
            # Ground-truth image
            # ----------------------------------------------------------

            if "gt" in visuals:

                gt_image = tensor2img(
                    [visuals["gt"]]
                )

                metric_data["img2"] = gt_image

                del self.gt

            # ----------------------------------------------------------
            # Release GPU memory
            # ----------------------------------------------------------

            del self.lq
            del self.output

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ----------------------------------------------------------
            # Save super-resolved image
            # ----------------------------------------------------------

            if save_img:

                if self.opt.get("is_train", False):

                    save_path = osp.join(
                        self.opt["path"]["visualization"],
                        image_name,
                        f"{image_name}_{current_iter}.png",
                    )

                else:

                    suffix = validation_config.get(
                        "suffix",
                        "",
                    )

                    if suffix:

                        filename = (
                            f"{image_name}_{suffix}.png"
                        )

                    else:

                        filename = (
                            f"{image_name}_"
                            f"{self.opt['name']}.png"
                        )

                    save_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        filename,
                    )

                imwrite(
                    sr_image,
                    save_path,
                )

            # ----------------------------------------------------------
            # Calculate validation metrics
            # ----------------------------------------------------------

            if calculate_metrics:

                for metric_name, metric_options in (
                    metrics_config.items()
                ):

                    metric_value = calculate_metric(
                        metric_data,
                        metric_options,
                    )

                    self.metric_results[
                        metric_name
                    ] += metric_value

            # ----------------------------------------------------------
            # Update progress
            # ----------------------------------------------------------

            if progress is not None:

                progress.update(1)

                progress.set_description(
                    f"Test: {image_name}"
                )

        # --------------------------------------------------------------
        # Close progress bar
        # --------------------------------------------------------------

        if progress is not None:
            progress.close()

        # --------------------------------------------------------------
        # Finalize metrics
        # --------------------------------------------------------------

        if calculate_metrics:

            total_images = index + 1

            for metric_name in self.metric_results.keys():

                self.metric_results[
                    metric_name
                ] /= total_images

                self._update_best_metric_result(
                    dataset_name,
                    metric_name,
                    self.metric_results[metric_name],
                    current_iter,
                )

            self._log_validation_metric_values(
                current_iter,
                dataset_name,
                tb_logger,
            )