import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


class WatermarkRemover:
    def __init__(self, inpaint_radius=3, sensitivity=1.0, method="auto",
                 dark=False, debug=False):
        """
        Parameters
        ----------
        inpaint_radius : int
            Radius passed to cv2.inpaint (larger = smoother fill, slower).
        sensitivity : float
            Multiplier on detection thresholds. >1 detects more aggressively,
            <1 detects less aggressively. Useful when the auto threshold misses
            a faint watermark or grabs too much background.
        method : str
            'telea'  – fast, good for thin strokes (text).
            'ns'     – Navier-Stokes, better for larger solid regions.
            'auto'   – blend TELEA and NS; weight is decided per-component by
                       size (small components favour TELEA, large favour NS).
        dark : bool
            Also detect dark/black semi-transparent watermarks. Off by default
            because it can produce false positives on naturally dark areas
            (shadowed rocks, dark foliage, etc.).
        debug : bool
            When True, writes debug_mask.png showing the detected watermark mask.
        """
        self.inpaint_radius = inpaint_radius
        self.sensitivity = sensitivity
        self.method = method
        self.dark = dark
        self.debug = debug

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_watermark(self, img):
        """Return a uint8 mask (255 = watermark) via HSV-based detection.

        Handles both *light* watermarks (white/gray semi-transparent overlays)
        and *dark* watermarks (black semi-transparent overlays).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        scale = min(img.shape[0], img.shape[1]) / 500.0

        def ksize(n):
            s = max(3, int(n * scale) | 1)
            return (s, s)

        # Local neighbourhood saturation (captures the desaturation dip caused
        # by a semi-transparent white/gray overlay on a colourful background).
        local_sat = cv2.GaussianBlur(sat, ksize(31), 0)
        local_val = cv2.GaussianBlur(val, ksize(31), 0)

        # Adaptive ceiling: 70th percentile of image saturation.
        sat_ceil = min(150.0, float(np.percentile(sat, 70)) * self.sensitivity)

        # ---- Light watermark: low-sat, high-val, local dip in saturation ----
        sat_diff_thresh = 40.0 / self.sensitivity
        light_mask = (
            (sat < sat_ceil) &
            (val > 80) &
            (local_sat - sat > sat_diff_thresh)
        ).astype(np.uint8) * 255

        if self.dark:
            # Dark watermark: low-sat, low-val, local dip in value.
            # Kept opt-in because naturally dark areas (shadowed rocks, foliage)
            # have the same signature and cause false positives by default.
            val_floor = max(30.0, float(np.percentile(val, 30)))
            val_diff_thresh = 30.0 / self.sensitivity
            dark_mask = (
                (sat < sat_ceil) &
                (val < val_floor) &
                (val - local_val < -val_diff_thresh)
            ).astype(np.uint8) * 255
            combined = cv2.bitwise_or(light_mask, dark_mask)
        else:
            combined = light_mask

        # Morphological cleanup: close small gaps within strokes, remove noise.
        k1 = ksize(5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Merge nearby detections into coherent blobs (kept small so that
        # TELEA/NS only has to fill individual strokes, not one giant rectangle).
        merge_w = max(5, int(10 * scale) | 1)
        merge_h = max(5, int(10 * scale) | 1)
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_w, merge_h))
        merged = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, merge_kernel)

        labeled, num_features = ndimage.label(merged)
        if num_features == 0:
            return merged

        # Keep components in a plausible size range.
        pixel_counts = np.bincount(labeled.ravel())[1:]  # skip background (label 0)
        min_size = max(30, int(img.shape[0] * img.shape[1] * 0.0005))
        max_size = int(img.shape[0] * img.shape[1] * 0.40)

        clean_mask = np.zeros_like(merged)
        for i, count in enumerate(pixel_counts):
            if min_size <= count <= max_size:
                clean_mask[labeled == (i + 1)] = 255

        # Dilate to catch semi-transparent edges (but never fill entire bounding box).
        dil_size = max(5, int(10 * scale) | 1)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
        final_mask = cv2.dilate(clean_mask, dil_kernel, iterations=2)

        return final_mask

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------

    def _inpaint(self, img, mask):
        """Run inpainting according to self.method.

        'auto' blends TELEA and NS, weighting each component by its size:
        small components (text strokes) favour TELEA; large blobs favour NS.
        """
        if self.method == "telea":
            return cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        if self.method == "ns":
            return cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_NS)

        # --- 'auto': per-component blending ---
        result_telea = cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        result_ns    = cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_NS)

        labeled, num_features = ndimage.label(mask > 0)
        if num_features == 0:
            return result_telea

        # Build a per-pixel blend weight map: weight_ns ∈ [0, 1]
        # Small components → weight_ns ≈ 0 (pure TELEA)
        # Large components → weight_ns ≈ 1 (pure NS)
        weight_ns = np.zeros(mask.shape, dtype=np.float32)

        pixel_counts = np.bincount(labeled.ravel())[1:]
        for i, count in enumerate(pixel_counts):
            # Sigmoid-like transition: 0 at 500px, 1 at 5000px
            t = np.clip((count - 500) / 4500.0, 0.0, 1.0)
            weight_ns[labeled == (i + 1)] = t

        weight_ns = weight_ns[:, :, np.newaxis]  # broadcast over channels
        blended = (
            result_telea.astype(np.float32) * (1 - weight_ns) +
            result_ns.astype(np.float32)    * weight_ns
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Texture refinement
    # ------------------------------------------------------------------

    def _texture_transfer(self, inpainted, original, mask):
        """Replace the smooth inpaint fill with texture-aware content.

        Strategy
        --------
        1. Frequency split: low-freq (colour/tone) vs high-freq (texture/grain).
        2. Low-freq  → keep from inpainted result (captured structure well).
        3. High-freq → copy from the nearest known pixel outside the mask.
        4. Inject *locally-estimated* noise so the fill has the same grain
           character as the specific region surrounding the watermark.
        """
        mask_bool = mask > 0

        # 1. Frequency decomposition
        blur_k = (15, 15)
        orig_lo  = cv2.GaussianBlur(original.astype(np.float32),  blur_k, 0)
        paint_lo = cv2.GaussianBlur(inpainted.astype(np.float32), blur_k, 0)
        orig_hi  = original.astype(np.float32) - orig_lo

        # 2. Nearest known pixel for each masked pixel
        _, nearest_idx = ndimage.distance_transform_edt(mask_bool, return_indices=True)
        nr = nearest_idx[0]
        nc = nearest_idx[1]

        # 3. Transplant high-freq texture from nearest known source
        transplanted_hi = orig_hi[nr, nc]
        result = paint_lo + transplanted_hi
        result = np.clip(result, 0, 255).astype(np.float32)

        # 4. Local noise estimation
        # Build a slightly dilated ring around the mask to sample nearby grain
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        nearby = cv2.dilate(mask, ring_kernel) & ~mask  # ring just outside mask
        nearby_bool = nearby > 0

        if nearby_bool.any():
            local_grain_std = float(np.std(orig_hi[nearby_bool]))
        else:
            local_grain_std = float(np.std(orig_hi[~mask_bool]))

        noise = np.random.normal(0, local_grain_std * 0.6, result.shape).astype(np.float32)
        result[mask_bool] += noise[mask_bool]
        result = np.clip(result, 0, 255)

        # 5. Replace only the masked region
        out = inpainted.copy().astype(np.float32)
        out[mask_bool] = result[mask_bool]
        return out.astype(np.uint8)

    def _smooth_boundary(self, result, mask):
        """Feather a narrow seam at the mask boundary for a natural edge.

        Only blends within the filled result so the watermark can never leak back.
        """
        weight = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0) / 255.0
        weight = weight[:, :, np.newaxis]

        blurred = cv2.GaussianBlur(result, (5, 5), 1.5).astype(np.float32)
        out = result.astype(np.float32) * (1 - weight * 0.35) + blurred * (weight * 0.35)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove_watermark(self, img, mask):
        """Three-pass removal: inpaint → texture refinement → boundary feather."""
        inpainted = self._inpaint(img, mask)
        result    = self._texture_transfer(inpainted, img, mask)
        result    = self._smooth_boundary(result, mask)
        return result

    def process(self, image_path, output_path="cleaned.png"):
        """Load *image_path*, remove watermark, write to *output_path*.

        Alpha channels are preserved for PNG files.
        """
        img_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"Cannot load {image_path}")

        if img_raw.ndim == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

        alpha = None
        if img_raw.shape[2] == 4:
            alpha = img_raw[:, :, 3]
            img = img_raw[:, :, :3]
        else:
            img = img_raw

        mask   = self.detect_watermark(img)
        result = self.remove_watermark(img, mask)

        if alpha is not None:
            result = np.dstack([result, alpha])

        cv2.imwrite(str(output_path), result)
        if self.debug:
            debug_path = Path(output_path).with_name("debug_mask.png")
            cv2.imwrite(str(debug_path), mask)
            print(f"Debug mask saved to {debug_path}")

        return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Remove semi-transparent watermarks from images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="IMAGE",
        help="Input image path(s). Accepts multiple files for batch processing.",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output path. For a single input, defaults to 'clean.<ext>'. "
            "For multiple inputs, use a directory path (created if needed)."
        ),
    )
    p.add_argument(
        "-r", "--radius",
        type=int,
        default=3,
        metavar="N",
        help="Inpainting radius (larger = smoother fill but slower).",
    )
    p.add_argument(
        "-s", "--sensitivity",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "Detection sensitivity multiplier. >1 catches more (risk: false positives), "
            "<1 catches less (risk: missed watermark)."
        ),
    )
    p.add_argument(
        "-m", "--method",
        choices=["telea", "ns", "auto"],
        default="auto",
        help=(
            "Inpainting method. 'auto' blends TELEA (good for thin text) and "
            "NS (good for large blobs) per component."
        ),
    )
    p.add_argument(
        "--dark",
        action="store_true",
        help=(
            "Also detect dark/black semi-transparent watermarks. "
            "Off by default to avoid false positives on naturally dark image areas."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save a debug_mask.png alongside each output showing the detected watermark.",
    )
    return p


def resolve_output(inputs, output_arg):
    """Return a list of (input_path, output_path) pairs."""
    inputs = [Path(p) for p in inputs]

    if len(inputs) == 1:
        src = inputs[0]
        if output_arg is None:
            dst = src.with_name(f"clean{src.suffix}")
        else:
            dst = Path(output_arg)
        return [(src, dst)]

    # Batch mode: output must be a directory (or None → same directory as input)
    if output_arg is None:
        out_dir = None
    else:
        out_dir = Path(output_arg)
        out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for src in inputs:
        if out_dir is None:
            dst = src.with_name(f"clean_{src.name}")
        else:
            dst = out_dir / f"clean_{src.name}"
        pairs.append((src, dst))
    return pairs


def main():
    parser = build_parser()
    args = parser.parse_args()

    remover = WatermarkRemover(
        inpaint_radius=args.radius,
        sensitivity=args.sensitivity,
        method=args.method,
        dark=args.dark,
        debug=args.debug,
    )

    pairs = resolve_output(args.inputs, args.output)

    for src, dst in pairs:
        print(f"Processing {src} → {dst} ...")
        try:
            remover.process(src, dst)
            print(f"  Saved {dst}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR processing {src}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
