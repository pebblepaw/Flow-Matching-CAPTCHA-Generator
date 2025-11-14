"""
Color-based region tokenization for CAPTCHA images.

Extracts colored foreground patches as tokens, ignoring white background and black noise.
"""

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import label as scipy_label


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
        max_aspect_ratio: float = 8.0
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

    def extract_connected_components(
        self,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
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

    def _merge_overlapping_boxes(
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
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        regions = []
        for x, y, w, h in bboxes:
            region = image[y:y+h, x:x+w, :]

            if self.normalize_regions and region.shape[0] > 0:
                current_h, current_w = region.shape[:2]
                scale = self.target_height / current_h
                target_w = max(1, int(current_w * scale))
                region = cv2.resize(region, (target_w, self.target_height))

            regions.append(region)

        return regions

    def tokenize(self, image: np.ndarray, return_mask: bool = False) -> list[MatLike] | tuple[list[MatLike], MatLike]:
        fg_mask = self.create_foreground_mask(image)
        bboxes = self.extract_connected_components(fg_mask)

        bboxes = self._split_different_color_tokens(image, bboxes, fg_mask)
        bboxes = self._merge_same_color_tokens(image, bboxes)

        regions = self.extract_region_images(image, bboxes)

        if return_mask:
            return regions, fg_mask
        return regions