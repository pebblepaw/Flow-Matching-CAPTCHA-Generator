"""
Color-based region tokenization for CAPTCHA images.

Extracts colored foreground patches as tokens, ignoring white background and black noise.
"""

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import label as scipy_label
import matplotlib.pyplot as plt


class ColorRegionTokenizer:

    def __init__(
        self,
        white_threshold: int = 200,
        black_threshold: int = 50,
        min_region_area: int = 45,
        max_region_area: int = 10000,
        target_height: int = 80,
        normalize_regions: bool = True,
        padding: int = 5,
        min_saturation: int = 15,
        max_aspect_ratio: float = 8.0,
        split_wide_regions: bool = True,
        color_similarity_threshold: float = 40
    ):
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold
        self.min_region_area = min_region_area
        self.max_region_area = max_region_area
        self.target_height = target_height
        self.normalize_regions = normalize_regions
        self.padding = padding
        self.min_saturation = min_saturation
        self.max_aspect_ratio = max_aspect_ratio
        self.split_wide_regions = split_wide_regions
        self.color_similarity_threshold = color_similarity_threshold

    def create_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        is_white = (v > 240) & (s < 10)
        is_black_noise = (gray < 20) & (r < 25) & (g < 25) & (b < 25) & (s < 5)

        foreground_mask = ~is_white & ~is_black_noise
        has_color = s > 10
        color_mask = foreground_mask & has_color

        is_grayscale_char = (gray < 220) & (gray > 30) & ~is_white & (s < 30)
        is_faint_char = (gray < 240) & (gray > 30) & ~is_white & (s >= 30)
        foreground_mask = color_mask | is_faint_char | is_grayscale_char

        kernel_tiny = np.ones((2, 2), np.uint8)
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        foreground_mask = cv2.dilate(foreground_mask.astype(np.uint8), kernel_cross, iterations=1)
        foreground_mask = cv2.erode(foreground_mask, kernel_tiny, iterations=1)

        return foreground_mask.astype(bool)

    def _remove_thin_connections(self, mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) == 0:
            return mask

        kernel_vert = np.ones((3, 1), np.uint8)
        eroded = cv2.erode(mask, kernel_vert, iterations=1)

        if np.sum(eroded) < np.sum(mask) * 0.5:
            kernel_small = np.ones((2, 1), np.uint8)
            eroded = cv2.erode(mask, kernel_small, iterations=1)

        if np.sum(eroded) > 0:
            labeled, num = scipy_label(eroded)
            result = np.zeros_like(mask)
            for label_id in range(1, num + 1):
                component = (labeled == label_id).astype(np.uint8)
                dilated_comp = cv2.dilate(component, kernel_vert, iterations=1)
                result = result | (dilated_comp & mask)
            return result
        else:
            return mask

    def _remove_thin_bridges(self, mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) == 0:
            return mask

        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)

        if np.sum(eroded) == 0:
            return mask

        labeled, num = scipy_label(eroded)
        if num == 0:
            return mask

        _, original_num = scipy_label(mask)
        if num <= original_num:
            eroded = cv2.erode(mask, kernel, iterations=2)
            if np.sum(eroded) == 0:
                return mask
            labeled, num = scipy_label(eroded)
            if num <= original_num:
                return mask

        result = np.zeros_like(mask)
        for label_id in range(1, num + 1):
            component = (labeled == label_id).astype(np.uint8)
            dilated = cv2.dilate(component, kernel, iterations=1)
            result = result | (dilated & mask)

        return result.astype(bool)

    def _color_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate Euclidean distance between two colors in RGB space."""
        return np.linalg.norm(color1.astype(float) - color2.astype(float))

    def _get_similar_color_mask(
        self,
        image: np.ndarray,
        seed_color: np.ndarray,
        fg_mask: np.ndarray,
        color_threshold: float = 40
    ) -> np.ndarray:
        """Create a mask for pixels with similar color to the seed (vectorized)."""
        # Vectorized color distance calculation
        color_diff = np.linalg.norm(image.astype(float) - seed_color.astype(float), axis=2)
        color_mask = (color_diff < color_threshold) & fg_mask
        return color_mask

    def remove_secondary_colors(
        self,
        token: np.ndarray,
        white_value: int = 255,
        hue_threshold: int = 5,
        value_threshold: int = 100
    ) -> np.ndarray:
        """
        Remove secondary colors from a token, keeping only the most dominant color.

        Strategy:
        1. Find all non-white pixels
        2. Find the most common color (by pixel count)
        3. Remove pixels that differ from dominant color

        Uses OR logic:
        - Different hue OR different brightness → remove pixel

        Args:
            token: Input token image (RGB)
            white_value: Threshold for considering a pixel as white/background (default: 255)
            hue_threshold: Max difference in HSV Hue channel (default: 5, out of 180)
            value_threshold: Max difference in HSV Value channel (default: 100, out of 255)

        Returns:
            Cleaned token with only the dominant color preserved
        """
        h, w = token.shape[:2]
        result = token.copy()

        # Create foreground mask (non-white pixels)
        gray = cv2.cvtColor(token, cv2.COLOR_RGB2GRAY)
        fg_mask = gray < (white_value - 10)

        if np.sum(fg_mask) == 0:
            return result

        # Get all foreground pixels
        fg_pixels = token[fg_mask]

        if len(fg_pixels) == 0:
            return result

        # Find the most common color by clustering similar colors
        # We need to group similar colors first, then pick the most common group
        visited = np.zeros(len(fg_pixels), dtype=bool)
        color_clusters = []

        # Simple color clustering with a generous threshold (group similar colors)
        color_similarity_threshold = 40  # RGB distance

        for i in range(len(fg_pixels)):
            if visited[i]:
                continue

            pixel_color = fg_pixels[i]

            # Find all pixels with similar color (vectorized)
            distances = np.linalg.norm(fg_pixels.astype(float) - pixel_color.astype(float), axis=1)
            similar = distances < color_similarity_threshold

            # Mark as visited
            visited[similar] = True

            # Get the cluster's representative color (median)
            cluster_pixels = fg_pixels[similar]
            cluster_color = np.median(cluster_pixels, axis=0).astype(np.uint8)

            # Count pixels in this cluster
            pixel_count = np.sum(similar)

            color_clusters.append((cluster_color, pixel_count))

        if len(color_clusters) == 0:
            return result

        # Sort by pixel count - largest cluster is the dominant color
        color_clusters.sort(key=lambda x: x[1], reverse=True)
        dominant_color = color_clusters[0][0]

        # Convert dominant color to HSV to get its Hue and Value
        dominant_hsv = cv2.cvtColor(dominant_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
        dominant_hue = dominant_hsv[0]
        dominant_saturation = dominant_hsv[1]
        dominant_value = dominant_hsv[2]

        # Convert entire token to HSV
        token_hsv = cv2.cvtColor(token, cv2.COLOR_RGB2HSV)
        token_hue = token_hsv[:, :, 0]
        token_saturation = token_hsv[:, :, 1]
        token_value = token_hsv[:, :, 2]

        # For very low saturation colors (near grayscale), don't filter by hue
        # because hue is unreliable for grayscale colors
        is_grayscale_dominant = dominant_saturation < 30

        if is_grayscale_dominant:
            # For grayscale dominant color, keep all low-saturation pixels
            # Also check value similarity for brightness matching
            value_diff = np.abs(token_value.astype(float) - dominant_value)
            dissimilar_mask = ((token_saturation > 50) | (value_diff > value_threshold)) & fg_mask
        else:
            # For colored pixels, filter by both Hue AND Value similarity
            # Handle circular nature of Hue (0 and 180 are close in color wheel)
            hue_diff = np.abs(token_hue.astype(float) - dominant_hue)
            # Wrap around for circular hue distance
            hue_diff = np.minimum(hue_diff, 180 - hue_diff)

            # Check value difference (brightness)
            value_diff = np.abs(token_value.astype(float) - dominant_value)

            # Remove if EITHER hue is too different OR value is too different
            dissimilar_mask = ((hue_diff > hue_threshold) | (value_diff > value_threshold)) & fg_mask

        # Set dissimilar pixels to white
        result[dissimilar_mask] = [255, 255, 255]

        return result

    def extract_connected_components_by_color(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """Extract connected components by grouping similar colors using flood fill."""
        structure = np.ones((3, 3), dtype=bool)
        labeled_mask, num_components = scipy_label(mask.astype(bool), structure=structure)

        components_info = []

        for component_id in range(1, num_components + 1):
            component_mask = labeled_mask == component_id
            rows, cols = np.where(component_mask)

            if len(rows) == 0:
                continue

            # Get the dominant color of this component
            component_pixels = image[component_mask]

            # Filter out near-white and near-black pixels for color calculation
            hsv_pixels = cv2.cvtColor(component_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            valid_color_mask = (hsv_pixels[:, 1] > 20) & (hsv_pixels[:, 2] > 30) & (hsv_pixels[:, 2] < 240)

            if np.sum(valid_color_mask) > 0:
                dominant_color = np.median(component_pixels[valid_color_mask], axis=0)
            else:
                dominant_color = np.median(component_pixels, axis=0)

            # Now flood fill from this component using similar colors
            similar_color_mask = self._get_similar_color_mask(image, dominant_color, mask, color_threshold=self.color_similarity_threshold)

            # Intersect with the foreground mask
            color_component_mask = similar_color_mask & mask

            # Label the color-based mask to get connected components
            labeled_color, num_color_comps = scipy_label(color_component_mask, structure=structure)

            for color_comp_id in range(1, num_color_comps + 1):
                comp_mask = labeled_color == color_comp_id
                area = np.sum(comp_mask)

                if area < self.min_region_area or area > self.max_region_area:
                    continue

                comp_rows, comp_cols = np.where(comp_mask)
                if len(comp_rows) == 0:
                    continue

                y_min, y_max = comp_rows.min(), comp_rows.max()
                x_min, x_max = comp_cols.min(), comp_cols.max()

                width = x_max - x_min + 1
                height = y_max - y_min + 1

                if height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio > self.max_aspect_ratio:
                        continue

                h, w = mask.shape
                y_min_pad = max(0, y_min - self.padding)
                y_max_pad = min(h - 1, y_max + self.padding)
                x_min_pad = max(0, x_min - self.padding)
                x_max_pad = min(w - 1, x_max + self.padding)

                width_pad = x_max_pad - x_min_pad
                height_pad = y_max_pad - y_min_pad

                components_info.append((x_min_pad, y_min_pad, width_pad, height_pad, dominant_color))

        # Remove duplicates (same bounding box)
        unique_components = []
        seen_boxes = set()
        for comp in components_info:
            box = comp[:4]
            if box not in seen_boxes:
                seen_boxes.add(box)
                unique_components.append(comp)

        unique_components.sort(key=lambda b: b[0])
        return unique_components

    def extract_connected_components(
        self,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Original connected component extraction (kept for compatibility)."""
        structure = np.ones((3, 3), dtype=bool)
        labeled_mask, num_components = scipy_label(mask.astype(bool), structure=structure)

        bboxes = []
        for component_id in range(1, num_components + 1):
            component_mask = labeled_mask == component_id
            rows, cols = np.where(component_mask)

            if len(rows) > 0:
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                area = np.sum(component_mask)
                if area < self.min_region_area or area > self.max_region_area:
                    continue

                width = x_max - x_min + 1
                height = y_max - y_min + 1

                if height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio > self.max_aspect_ratio:
                        continue

                h, w = mask.shape
                y_min = max(0, y_min - self.padding)
                y_max = min(h - 1, y_max + self.padding)
                x_min = max(0, x_min - self.padding)
                x_max = min(w - 1, x_max + self.padding)

                width = x_max - x_min
                height = y_max - y_min
                bboxes.append((x_min, y_min, width, height))

        bboxes.sort(key=lambda b: b[0])
        return bboxes

    def _split_wide_regions(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        result = []

        for x, y, w, h in bboxes:
            if w > 1.5 * h:
                region_mask = mask[y:y+h, x:x+w].astype(np.uint8)
                split_boxes = self._try_erosion_split(region_mask, x, y, w, h)
                if split_boxes and len(split_boxes) > 1:
                    result.extend(split_boxes)
                else:
                    result.append((x, y, w, h))
            else:
                result.append((x, y, w, h))

        return result

    def _try_erosion_split(
        self,
        region_mask: np.ndarray,
        base_x: int,
        base_y: int,
        w: int,
        h: int
    ) -> List[Tuple[int, int, int, int]]:
        kernel = np.ones((4, 1), np.uint8)
        eroded = cv2.erode(region_mask, kernel, iterations=3)
        labeled, num = scipy_label(eroded)

        if num <= 1:
            return []

        boxes = []
        for label_id in range(1, num + 1):
            component = labeled == label_id
            rows, cols = np.where(component)

            if len(rows) > 0:
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                seg_w = x_max - x_min + 1
                seg_h = y_max - y_min + 1

                if seg_w > h * 0.15 and seg_h > h * 0.3:
                    boxes.append((base_x + x_min, base_y + y_min, seg_w, seg_h))

        boxes.sort(key=lambda b: b[0])
        return boxes if len(boxes) > 1 else []

    def _try_projection_split(
        self,
        region_mask: np.ndarray,
        base_x: int,
        base_y: int,
        w: int,
        h: int
    ) -> List[Tuple[int, int, int, int]]:
        projection = np.sum(region_mask, axis=0)
        thresholds = [
            np.mean(projection[projection > 0]) * 0.15,
            np.max(projection) * 0.10,
            np.percentile(projection[projection > 0], 25) * 0.5
        ]

        for threshold in thresholds:
            valleys = projection < threshold
            split_points = []
            in_valley = False
            valley_start = 0

            for i, is_valley in enumerate(valleys):
                if is_valley and not in_valley:
                    valley_start = i
                    in_valley = True
                elif not is_valley and in_valley:
                    valley_width = i - valley_start
                    if valley_width >= 2:
                        valley_mid = (valley_start + i) // 2
                        split_points.append(valley_mid)
                    in_valley = False

            if split_points:
                boxes = []
                prev_x = 0

                for split_x in split_points + [w]:
                    segment_width = split_x - prev_x
                    if segment_width > h * 0.15:
                        boxes.append((base_x + prev_x, base_y, segment_width, h))
                    prev_x = split_x

                if len(boxes) > 1:
                    return boxes

        return []

    def _try_gradient_split(
        self,
        region_mask: np.ndarray,
        base_x: int,
        base_y: int,
        w: int,
        h: int
    ) -> List[Tuple[int, int, int, int]]:
        projection = np.sum(region_mask, axis=0)
        gradient = np.diff(projection, prepend=projection[0])
        smoothed_grad = np.convolve(gradient, np.ones(3)/3, mode='same')
        threshold = -np.std(smoothed_grad) * 0.5
        neg_gradients = smoothed_grad < threshold
        split_candidates = np.where(neg_gradients)[0]

        if len(split_candidates) == 0:
            return []

        split_points = []
        last_split = -999

        for candidate in split_candidates:
            if candidate - last_split > h * 0.3:
                split_points.append(candidate)
                last_split = candidate

        if split_points:
            boxes = []
            prev_x = 0

            for split_x in split_points + [w]:
                segment_width = split_x - prev_x
                if segment_width > h * 0.15:
                    boxes.append((base_x + prev_x, base_y, segment_width, h))
                prev_x = split_x

            if len(boxes) > 1:
                return boxes

        return []

    def merge_overlapping_boxes(
        self,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        if len(bboxes) <= 1:
            return bboxes

        merged = []
        current = list(bboxes[0])

        for box in bboxes[1:]:
            x, y, w, h = box
            curr_x, curr_y, curr_w, curr_h = current
            overlap = False
            curr_right = curr_x + curr_w
            box_right = x + w

            if not (x > curr_right or curr_x > box_right):
                curr_bottom = curr_y + curr_h
                box_bottom = y + h

                if not (y > curr_bottom or curr_y > box_bottom):
                    overlap = True

            if overlap:
                new_x = min(curr_x, x)
                new_y = min(curr_y, y)
                new_right = max(curr_right, box_right)
                new_bottom = max(curr_bottom, box_bottom)
                current = [new_x, new_y, new_right - new_x, new_bottom - new_y]
            else:
                merged.append(tuple(current))
                current = list(box)

        merged.append(tuple(current))
        return merged

    def _get_dominant_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w, :]

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]

        colored_mask = s > 20
        if np.sum(colored_mask) == 0:
            return np.array([128, 128, 128])

        colored_pixels = region[colored_mask]
        median_color = np.median(colored_pixels, axis=0)
        return median_color

    def _colors_match(self, color1: np.ndarray, color2: np.ndarray, threshold: float = 30) -> bool:
        distance = np.linalg.norm(color1 - color2)
        return distance < threshold

    def _merge_same_color_tokens(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        if len(bboxes) <= 1:
            return bboxes

        merged = []
        i = 0
        while i < len(bboxes):
            current_box = bboxes[i]
            current_color = self._get_dominant_color(image, current_box)

            if i + 1 < len(bboxes):
                next_box = bboxes[i + 1]
                next_color = self._get_dominant_color(image, next_box)

                x1, y1, w1, h1 = current_box
                x2, y2, w2, h2 = next_box

                horizontal_gap = x2 - (x1 + w1)
                vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
                height_diff = abs(h1 - h2)

                min_width = min(w1, w2)
                gap_threshold = min(5, min_width * 0.15)

                is_aligned = vertical_overlap > min(h1, h2) * 0.7
                is_similar_height = height_diff < max(h1, h2) * 0.3
                is_very_similar_color = np.linalg.norm(current_color - next_color) < 10

                is_vertically_stacked = False
                if horizontal_gap < 0:
                    horizontal_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                    if horizontal_overlap > min_width * 0.5 and vertical_overlap < min(h1, h2) * 0.5:
                        is_vertically_stacked = True

                if is_vertically_stacked and is_very_similar_color:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_right = max(x1 + w1, x2 + w2)
                    new_bottom = max(y1 + h1, y2 + h2)
                    merged.append((new_x, new_y, new_right - new_x, new_bottom - new_y))
                    i += 2
                    continue

                if (horizontal_gap < gap_threshold and horizontal_gap >= -5 and
                    is_very_similar_color and
                    is_aligned and is_similar_height):
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_right = max(x1 + w1, x2 + w2)
                    new_bottom = max(y1 + h1, y2 + h2)
                    merged.append((new_x, new_y, new_right - new_x, new_bottom - new_y))
                    i += 2
                    continue

            merged.append(current_box)
            i += 1

        return merged

    def _split_different_color_tokens(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        fg_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        result = []

        for bbox in bboxes:
            x, y, w, h = bbox
            region = image[y:y+h, x:x+w, :]
            region_mask = fg_mask[y:y+h, x:x+w]

            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hue = hsv_region[:, :, 0]
            sat = hsv_region[:, :, 1]

            colored_mask = (sat > 20) & region_mask
            if np.sum(colored_mask) < 50:
                result.append(bbox)
                continue

            rows, cols = np.where(colored_mask)
            if len(rows) < 10:
                result.append(bbox)
                continue

            colored_pixels = region[colored_mask]
            hue_values = hue[colored_mask]

            quantized_hues = (hue_values // 30).astype(int)
            unique_hues, counts = np.unique(quantized_hues, return_counts=True)

            significant_hues = unique_hues[counts > len(hue_values) * 0.25]

            if len(significant_hues) < 2:
                result.append(bbox)
                continue

            hue_colors = []
            for h in significant_hues:
                mask_h = quantized_hues == h
                median_color = np.median(colored_pixels[mask_h], axis=0)
                hue_colors.append(median_color)

            distinct_hues = []
            for i, h in enumerate(significant_hues):
                is_distinct = True
                for j in distinct_hues:
                    if np.linalg.norm(hue_colors[i] - hue_colors[j]) < 80:
                        is_distinct = False
                        break
                if is_distinct:
                    distinct_hues.append(i)

            if len(distinct_hues) < 2:
                result.append(bbox)
                continue

            split_components = []
            for hue_idx in distinct_hues:
                target_hue = significant_hues[hue_idx]
                cluster_mask = np.zeros_like(colored_mask, dtype=bool)

                pixel_idx = 0
                for row, col in zip(rows, cols):
                    if quantized_hues[pixel_idx] == target_hue:
                        cluster_mask[row, col] = True
                    pixel_idx += 1

                structure = np.ones((3, 3), dtype=bool)
                labeled_cluster, num = scipy_label(cluster_mask, structure=structure)

                for comp_id in range(1, num + 1):
                    comp_mask = labeled_cluster == comp_id
                    area = np.sum(comp_mask)

                    if area < self.min_region_area * 2:
                        continue

                    comp_rows, comp_cols = np.where(comp_mask)
                    if len(comp_rows) == 0:
                        continue

                    cy_min, cy_max = comp_rows.min(), comp_rows.max()
                    cx_min, cx_max = comp_cols.min(), comp_cols.max()
                    comp_width = cx_max - cx_min + 1
                    comp_height = cy_max - cy_min + 1

                    if comp_width < 5 or comp_height < 5:
                        continue

                    split_components.append((x + cx_min, y + cy_min, comp_width, comp_height))

            if len(split_components) >= 2:
                should_split = False
                for i in range(len(split_components)):
                    x1, y1, w1, h1 = split_components[i]
                    for j in range(i + 1, len(split_components)):
                        x2, y2, w2, h2 = split_components[j]

                        left = max(x1, x2)
                        right = min(x1 + w1, x2 + w2)
                        h_overlap = max(0, right - left)

                        top = max(y1, y2)
                        bottom = min(y1 + h1, y2 + h2)
                        v_overlap = max(0, bottom - top)

                        has_both_overlap = h_overlap > min(w1, w2) * 0.5 and v_overlap > min(h1, h2) * 0.5
                        has_vertical_overlap = v_overlap > min(h1, h2) * 0.6 and h_overlap > 0

                        if has_both_overlap or has_vertical_overlap:
                            should_split = True
                            break

                    if should_split:
                        break

                if should_split:
                    result.extend(split_components)
                else:
                    result.append(bbox)
                continue

            result.append(bbox)

        result.sort(key=lambda b: b[0])
        return result

    def extract_region_images(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        remove_other_colors: bool = False
    ) -> List[np.ndarray]:
        """Extract region images, optionally removing pixels with different colors."""
        regions = []
        for bbox_info in bboxes:
            # Handle both (x, y, w, h) and (x, y, w, h, color) formats
            if len(bbox_info) == 5:
                x, y, w, h, dominant_color = bbox_info
            else:
                x, y, w, h = bbox_info
                dominant_color = None

            region = image[y:y+h, x:x+w, :].copy()

            # Remove other colors if requested and dominant color is available
            if remove_other_colors and dominant_color is not None:
                # Vectorized: create a mask for pixels similar to the dominant color
                color_diff = np.linalg.norm(region.astype(float) - dominant_color.astype(float), axis=2)
                keep_mask = color_diff < 40

                # Set non-matching pixels to white
                region[~keep_mask] = [255, 255, 255]

            if self.normalize_regions and region.shape[0] > 0:
                current_h, current_w = region.shape[:2]
                scale = self.target_height / current_h
                target_w = max(1, int(current_w * scale))
                region = cv2.resize(region, (target_w, self.target_height))

            regions.append(region)

        return regions

    def tokenize(self, image: np.ndarray, return_mask: bool = False, use_color_segmentation: bool = False, remove_other_colors: bool = False) -> list[MatLike] | tuple[list[MatLike], MatLike]:
        """
        Tokenize image into character regions.

        Args:
            image: Input RGB image
            return_mask: Whether to return foreground mask
            use_color_segmentation: If True, use color-based connected components with flood fill (experimental)
                                   If False (default), use original split/merge method (better performance)
            remove_other_colors: If True, remove pixels with different colors from each token (only used with color segmentation)
        """
        fg_mask = self.create_foreground_mask(image)

        if use_color_segmentation:
            # Use new color-based segmentation
            bboxes = self.extract_connected_components_by_color(image, fg_mask)
            # Extract regions with color filtering
            regions = self.extract_region_images(image, bboxes, remove_other_colors=remove_other_colors)
        else:
            # Use original method
            bboxes = self.extract_connected_components(fg_mask)
            bboxes = self._split_different_color_tokens(image, bboxes, fg_mask)
            bboxes = self._merge_same_color_tokens(image, bboxes)
            regions = self.extract_region_images(image, bboxes)

        if return_mask:
            return regions, fg_mask
        return regions

    def visualize_tokens(
        self,
        image: np.ndarray,
        tokens: List[np.ndarray],
        title: str = "Color Region Tokens",
        show_mask: bool = True
    ):
        n_tokens = len(tokens)
        n_rows = 3 if show_mask else 2
        fig, axes = plt.subplots(n_rows, max(n_tokens, 1),
                                 figsize=(2 * max(n_tokens, 1), 2 * n_rows))

        if n_tokens == 1:
            axes = axes.reshape(n_rows, 1)

        for ax in axes[0]:
            ax.axis('off')
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")

        if show_mask:
            for ax in axes[1]:
                ax.axis('off')
            fg_mask = self.create_foreground_mask(image)
            axes[1, 0].imshow(fg_mask, cmap='gray')
            axes[1, 0].set_title("Foreground Mask")

        token_row = 2 if show_mask else 1
        for i, token in enumerate(tokens):
            if i < n_tokens:
                axes[token_row, i].imshow(token)
                axes[token_row, i].set_title(f"Token {i+1}\n{token.shape[1]}×{token.shape[0]}")
                axes[token_row, i].axis('off')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def visualize_with_boxes(
        self,
        image: np.ndarray,
        tokens: List[np.ndarray]
    ):
        fg_mask = self.create_foreground_mask(image)
        bboxes = self.extract_connected_components(fg_mask)
        bboxes = self.merge_overlapping_boxes(bboxes)
        img_with_boxes = image.copy()

        for i, (x, y, w, h) in enumerate(bboxes):
            color = plt.cm.tab10(i % 10)[:3]
            color = tuple(int(c * 255) for c in color)
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_with_boxes, f"{i+1}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(img_with_boxes)
        axes[1].set_title(f"Detected Regions ({len(bboxes)} tokens)")
        axes[1].axis('off')
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    import os

    print("Color Region Tokenization Demo")
    print("=" * 70)

    tokenizer = ColorRegionTokenizer(
        white_threshold=200,
        black_threshold=50,
        min_region_area=100,
        max_region_area=10000,
        target_height=80,
        padding=5,
        min_saturation=15,
        max_aspect_ratio=8.0,
        split_wide_regions=True
    )

    train_dir = "data/raw/train"
    sample_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png')])[:5]

    for filename in sample_files:
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)
        label = filename.split('-')[0]
        tokens = tokenizer.tokenize(img)

        print(f"{filename} -> '{label}'")
        print(f"  Expected: {len(label)} characters")
        print(f"  Extracted: {len(tokens)} regions")
        print(f"  Shapes: {[f'{t.shape[1]}x{t.shape[0]}' for t in tokens]}")

        if sample_files.index(filename) < 3:
            fig1 = tokenizer.visualize_tokens(img, tokens,
                                             title=f"'{label}' - Color Regions",
                                             show_mask=True)
            plt.savefig(f"color_tokens_{label}.png", dpi=150, bbox_inches='tight')
            plt.close()

            fig2 = tokenizer.visualize_with_boxes(img, tokens)
            plt.savefig(f"color_boxes_{label}.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved visualizations")
        print()

    print("=" * 70)
    print("Complete!")
