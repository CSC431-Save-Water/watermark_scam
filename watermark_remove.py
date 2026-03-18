import sys
import cv2
import numpy as np
from scipy import ndimage


class WatermarkRemover:
    def __init__(self):
        self.inpaint_radius = 3

    def detect_watermark(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        scale = min(img.shape[0], img.shape[1]) / 500.0

        def ksize(n):
            s = max(3, int(n * scale) | 1)
            return (s, s)

        # Compare each pixel to a close neighborhood (~25px at 500px).
        # Watermarks are semi-transparent white/gray overlays: they desaturate
        # whatever background is beneath them, creating a local saturation dip
        # even on highly-saturated backgrounds (e.g. orange sky). A small kernel
        # captures this dip without averaging across the whole watermark region,
        # which would flatten the signal on strongly-coloured images.
        local_sat = cv2.GaussianBlur(sat, ksize(31), 0)

        # Adaptive absolute saturation ceiling: watermarks lift sat toward white/gray,
        # so sat stays below this even on colourful backgrounds.
        # Use 70th percentile of image saturation as adaptive ceiling.
        sat_ceil = min(150.0, float(np.percentile(sat, 70)))

        combined = (
            (sat < sat_ceil) &
            (val > 80) &
            (local_sat - sat > 40)
        ).astype(np.uint8) * 255

        k1 = ksize(5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Merge nearby detections within a single letter stroke, but keep the
        # horizontal gap small so individual letters stay separate blobs.
        # Large horizontal merging used to create one giant rectangle that
        # was impossible to inpaint well; keeping letters separate means TELEA
        # only has to fill thin strokes, which it handles cleanly.
        merge_w = max(5, int(10 * scale) | 1)
        merge_h = max(5, int(10 * scale) | 1)
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_w, merge_h))
        merged = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, merge_kernel)

        labeled, num_features = ndimage.label(merged)
        if num_features == 0:
            return merged

        # Keep every component large enough to plausibly be a watermark stroke.
        # Minimum size: ~0.5% of image pixels (avoids tiny noise blobs).
        pixel_counts = np.array([np.sum(labeled == (i + 1)) for i in range(num_features)])
        min_size = max(30, int(img.shape[0] * img.shape[1] * 0.0005))
        # Also discard components covering > 40% of image (likely false-positive background region)
        max_size = int(img.shape[0] * img.shape[1] * 0.40)

        clean_mask = np.zeros_like(merged)
        for i, count in enumerate(pixel_counts):
            if min_size <= count <= max_size:
                clean_mask[labeled == (i + 1)] = 255

        # Dilate slightly to catch semi-transparent edges — do NOT fill the bounding box.
        # Filling the whole bounding box forces inpainting over a huge solid rectangle,
        # which is why the old code produced blurry/blocky output.
        dil_size = max(5, int(10 * scale) | 1)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
        final_mask = cv2.dilate(clean_mask, dil_kernel, iterations=2)

        return final_mask

    def _texture_transfer(self, inpainted, original, mask):
        """
        Replace the smooth TELEA fill with texture-aware content.

        Strategy:
          1. Frequency split: separate low-freq (colour/tone) from high-freq (texture/grain).
          2. Low-freq  → keep from the inpainted result (TELEA captured structure well).
          3. High-freq → copy from the nearest *known* pixel outside the mask.
          4. Inject a small amount of matched noise so the filled area has the same
             grain character as the surrounding image, eliminating the "too clean" look.
        """
        mask_bool = mask > 0

        # --- 1. Frequency decomposition ---
        blur_k = (15, 15)
        orig_lo   = cv2.GaussianBlur(original.astype(np.float32),  blur_k, 0)
        paint_lo  = cv2.GaussianBlur(inpainted.astype(np.float32), blur_k, 0)

        # High-frequency detail layers
        orig_hi   = original.astype(np.float32)  - orig_lo   # texture/grain outside mask
        # (we don't compute inpaint_hi because TELEA smears it)

        # --- 2. Find nearest known pixel for every masked pixel ---
        _, nearest_idx = ndimage.distance_transform_edt(mask_bool, return_indices=True)
        nr = nearest_idx[0]  # row of nearest known pixel
        nc = nearest_idx[1]  # col  of nearest known pixel

        # --- 3. Transplant high-freq texture from the nearest known source ---
        transplanted_hi = orig_hi[nr, nc]   # shape (h, w, 3) – each pixel mapped

        # Recombine: smooth base from inpaint + texture from nearest known pixel
        result = paint_lo + transplanted_hi
        result = np.clip(result, 0, 255).astype(np.float32)

        # --- 4. Add matched noise ---
        # Estimate local grain amplitude just outside the mask (use the ring of nearest
        # known pixels so we match the *actual* noise level of the background there).
        grain_std = float(np.std(orig_hi[~mask_bool]))
        noise = np.random.normal(0, grain_std * 0.6, result.shape).astype(np.float32)
        result[mask_bool] += noise[mask_bool]
        result = np.clip(result, 0, 255)

        # --- 5. Replace only the masked region; leave the rest untouched ---
        out = inpainted.copy().astype(np.float32)
        out[mask_bool] = result[mask_bool]

        return out.astype(np.uint8)

    def _smooth_boundary(self, result, mask):
        """
        Blend a narrow seam at the mask boundary so the fill edges look natural.
        We only mix pixels from the *filled result* (not the original watermarked image),
        so the watermark never leaks back in.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        outer_edge = cv2.dilate(mask, kernel) - mask  # ring just outside the fill

        # Feather weight: 1 at mask edge, fades to 0 a few pixels out
        weight = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0) / 255.0
        weight = weight[:, :, np.newaxis]  # broadcast over channels

        # Lightly blur the result and mix only at the seam zone
        blurred = cv2.GaussianBlur(result, (5, 5), 1.5).astype(np.float32)
        out = result.astype(np.float32) * (1 - weight * 0.35) + blurred * (weight * 0.35)

        return np.clip(out, 0, 255).astype(np.uint8)

    def remove_watermark(self, img, mask):
        # Pass 1: TELEA captures smooth background structure quickly
        inpainted = cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        # Pass 2: texture-aware refinement + noise injection
        result = self._texture_transfer(inpainted, img, mask)

        # Pass 3: feather the seam (blend within the result only, never back into original)
        result = self._smooth_boundary(result, mask)

        return result

    def process(self, image_path, output_path="cleaned.png"):
        # Use IMREAD_UNCHANGED so alpha channel is preserved for PNGs
        img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"Cannot load {image_path}")

        # Split off alpha if present — inpainting only works on BGR
        if img_raw.ndim == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        alpha = None
        if img_raw.shape[2] == 4:
            alpha = img_raw[:, :, 3]
            img = img_raw[:, :, :3]
        else:
            img = img_raw

        mask = self.detect_watermark(img)
        result = self.remove_watermark(img, mask)

        # Reattach alpha channel if the source had one
        if alpha is not None:
            result = np.dstack([result, alpha])

        cv2.imwrite(output_path, result)
        cv2.imwrite("debug_mask.png", mask)
        return result


if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "watermarked.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "clean.png"
    remover = WatermarkRemover()
    remover.process(input_path, output_path)
