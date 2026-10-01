"""Pinned upstream core for reproducibility; see LICENSE and ATTRIBUTION.md.

Extracted without algorithmic edits from Emory-HITI/knee-crop revision
c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93, pipeline.py lines 112-436.
Only imports were adapted to this package. Do not use upstream success flags.
"""

import logging
from math import degrees
from typing import Optional

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .config import CropConfig

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return degrees(np.arccos(cos_angle))

def find_vertical_line(edge_region, x_offset, config: Optional[CropConfig] = None):
    cfg = config or CropConfig()
    try:
        y_coords, x_coords = np.where(edge_region > 0)
        if len(y_coords) == 0:
            return None
        
        x_coords += x_offset
        points = list(zip(x_coords, y_coords))
        points.sort(key=lambda p: p[1])
        
        current_segment = [points[0]]
        all_segments = []
        
        for i in range(1, len(points)):
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            
            if abs(curr_y - prev_y) <= 2 and abs(curr_x - prev_x) <= 2:
                if len(current_segment) > 2:
                    start_x, start_y = current_segment[0]
                    dx = curr_x - start_x
                    dy = curr_y - start_y
                    if dx != 0:
                        angle = abs(np.arctan(dy/dx)) * 180/np.pi
                        if angle < 60:
                            all_segments.append(current_segment)
                            current_segment = [(curr_x, curr_y)]
                            continue
                current_segment.append((curr_x, curr_y))
            else:
                if len(current_segment) > 400:
                    all_segments.append(current_segment)
                current_segment = [(curr_x, curr_y)]
        
        if len(current_segment) > 400:
            all_segments.append(current_segment)
        
        min_height = int(edge_region.shape[0] * cfg.min_vert_line_frac)
        valid_segments = [seg for seg in all_segments if len(seg) >= min_height]
        
        if valid_segments:
            return max(valid_segments, key=len)
    except Exception as e:
        print(f"Error in find_vertical_line: {str(e)}")
    return None

def find_horizontal_line(edge_region, y_offset, config: Optional[CropConfig] = None):
    cfg = config or CropConfig()
    try:
        y_coords, x_coords = np.where(edge_region > 0)
        if len(x_coords) == 0:
            return None
        
        y_coords += y_offset
        points = list(zip(x_coords, y_coords))
        points.sort(key=lambda p: p[0])
        
        current_segment = [points[0]]
        all_segments = []
        
        for i in range(1, len(points)):
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            
            if abs(curr_x - prev_x) <= 2 and abs(curr_y - prev_y) <= 2:
                if len(current_segment) > 2:
                    start_x, start_y = current_segment[0]
                    dx = curr_x - start_x
                    dy = curr_y - start_y
                    if dy != 0:
                        angle = abs(np.arctan(dx/dy)) * 180/np.pi
                        if angle > 60:
                            all_segments.append(current_segment)
                            current_segment = [(curr_x, curr_y)]
                            continue
                current_segment.append((curr_x, curr_y))
            else:
                if len(current_segment) > 400:
                    all_segments.append(current_segment)
                current_segment = [(curr_x, curr_y)]
        
        if len(current_segment) > 400:
            all_segments.append(current_segment)
        
        min_width = int(edge_region.shape[1] * cfg.min_horiz_line_frac)
        valid_segments = [seg for seg in all_segments if len(seg) >= min_width]
        
        if valid_segments:
            return max(valid_segments, key=len)
    except Exception as e:
        print(f"Error in find_horizontal_line: {str(e)}")
    return None

def get_black_box_rows(image, config: Optional[CropConfig] = None):
    cfg = config or CropConfig()
    # Check if image is valid
    if image is None:
        raise ValueError("The image is not loaded correctly.")
    
    # Step 1: Threshold to isolate zero-intensity pixels
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)  # 0s become 255, others 0
    
    # Step 2: Morphological closing to clean up small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_kernel))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Filter for rows with sharp angles
    black_box_rows = []
    selected_contours = []
    image_width = image.shape[1]
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > cfg.black_box_min_area_px:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
            num_sides = len(approx)
            if num_sides >= 3:
                points = approx.reshape(num_sides, 2)
                angles = []
                for j in range(num_sides):
                    p1 = points[j]
                    p2 = points[(j + 1) % num_sides]
                    p3 = points[(j + 2) % num_sides]
                    angle = calculate_angle(p1, p2, p3)
                    angles.append(angle)

                logging.info(f"Contour {i}: angles={angles}, num_sides={num_sides}, x={x}, y={y}, w={w}, h={h}")

                # Count how many angles are in the 80-100 range
                sharp_angle_count = sum(1 for angle in angles if 85 <= angle <= 95)
                if 3 <= sharp_angle_count <= 8:  # At least 3 angles must be sharp
                    selected_contours.append(approx)  # Store the selected contour
                    for j, angle in enumerate(angles):
                        if 80 <= angle <= 100:
                            row = points[(j + 1) % num_sides][1]
                            black_box_rows.append(row)
                            logging.info(f"Contour {i}, vertex {j}: angle={angle:.2f}, row={row}")
    
    # Remove duplicates and sort
    return sorted(list(set(black_box_rows)))

def get_mins(img_, peak_):
    col_sums = img_.sum(0)
    threshold = col_sums[peak_] * 0.2
    below = np.where(col_sums < threshold)[0]
    left = below[below < peak_][-1] if any(below < peak_) else np.argmin(col_sums[:peak_])
    right = below[below > peak_][0] if any(below > peak_) else peak_ + np.argmin(col_sums[peak_:])
    return left, right

def preprocess_and_crop(img, config: Optional[CropConfig] = None, tag: str = ""):
    cfg = config or CropConfig()
    col_sums = img.sum(0)
    row_sums = img.sum(1)
    valid_cols = np.where((col_sums > 0) & (col_sums < 254 * img.shape[0]))[0]
    valid_rows = np.where((row_sums > 0) & (row_sums < 254 * img.shape[1]))[0]

    if len(valid_rows) > 0 and len(valid_cols) > 0:
        img = img[valid_rows[0]:valid_rows[-1], valid_cols[0]:valid_cols[-1]]

    img = cv2.resize(img, (int(cfg.resize_height * img.shape[1] / img.shape[0]), cfg.resize_height))
    img = img[cfg.vertical_trim:-cfg.vertical_trim, :]

    try:
    # Edge detection for line finding
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude)
        edges = np.uint8(magnitude > cfg.sobel_threshold) * 255
    
        # Find lines in all regions
        height, width = img.shape
        left_region = edges[:, :int(width*cfg.line_search_frac)]
        right_region = edges[:, int(width*(1 - cfg.line_search_frac)):]
        top_region = edges[:int(height*cfg.line_search_frac), :]
        bottom_region = edges[int(height*(1 - cfg.line_search_frac)):, :]
    
        left_line  = find_vertical_line(left_region, 0, config=cfg)
        right_line = find_vertical_line(right_region, int(width * (1 - cfg.line_search_frac)), config=cfg)
        top_line   = find_horizontal_line(top_region, 0, config=cfg)
        bottom_line= find_horizontal_line(bottom_region, int(height * (1 - cfg.line_search_frac)), config=cfg)
    
        # Initialize crop coordinates with full image
        should_crop = False
        if any([left_line, right_line, top_line, bottom_line]):
            left = 0
            right = width
            top = 0
            bottom = height
    
            if left_line and len(left_line) > 0:
                left = max(x for x, y in left_line)
                should_crop = True
            
            if right_line and len(right_line) > 0:
                right = min(x for x, y in right_line)
                should_crop = True
            
            if top_line and len(top_line) > 0:
                top = max(y for x, y in top_line)
                should_crop = True
            
            if bottom_line and len(bottom_line) > 0:
                bottom = min(y for x, y in bottom_line)
                should_crop = True
    
            # Apply cropping only if valid lines were found
            if should_crop:
                try:
                    img = img[top:bottom, left:right]
                    logging.info(f"Image cropped with lines: top={top}, bottom={bottom}, left={left}, right={right}")
                except Exception as e:
                    logging.warning(f"Cropping failed: {str(e)}. Continuing with original image.")

    except Exception as e:
        logging.warning(f"Line detection failed for {tag}: {str(e)}. Continuing with original image.")
    return img

def process_knee_side(knee_img, is_right_knee=True, tag="temp", stage="", config: Optional[CropConfig] = None, return_geometry=False):
    cfg = config or CropConfig()
    intensity_sum = knee_img.sum(0)
    peak = cfg.intensity_offset + np.argmax(intensity_sum[cfg.intensity_offset:-cfg.intensity_offset]) if is_right_knee else np.argmax(intensity_sum[:-cfg.intensity_offset])
    left_min, right_min = get_mins(knee_img, peak)
    cropped_horizontal = knee_img[:, left_min:right_min]
    lateral = knee_img[:, :peak] if is_right_knee else knee_img[:, peak:]
    ptp = abs(right_min - left_min)
    crop_peak = np.argmax(cropped_horizontal.sum(0))
    portion = cropped_horizontal[:, :crop_peak] if is_right_knee else cropped_horizontal[:, crop_peak:]

    # === Updated CLAHE using OpenCV ===
    if portion.dtype != np.uint8:
        portion_scaled = ((portion - portion.min()) / (portion.max() - portion.min()) * 255).astype(np.uint8)
    else:
        portion_scaled = portion
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tile)
    clahe_img = clahe.apply(portion_scaled)

    row_sums = clahe_img.sum(1)
    updated = row_sums.copy()
    height = clahe_img.shape[0]
    start_row, end_row = int(cfg.start_row_frac*height), int(cfg.end_row_frac*height)

    # === Black Box Detection ===
    black_box_rows = get_black_box_rows(lateral, config=cfg)
    if black_box_rows:
        logging.info(f"Black box detected in {tag} {stage}")
        before = [r for r in black_box_rows if r < 0.4*height]
        after = [r for r in black_box_rows if r > 0.6*height]
        if before: start_row = max(start_row, max(before) + cfg.black_box_offset)
        if after: end_row = min(end_row, min(after) - cfg.black_box_offset)

    # === Edge Detection ===
    blurred = cv2.GaussianBlur(portion, (cfg.gaussian_kernel), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag_visual = cv2.convertScaleAbs(mag)
    row_sums_portion = portion.sum(axis=1)
    if np.any(row_sums_portion > 0):
        max_intensity_row_index = np.argmax(row_sums_portion)
    else:
        max_intensity_row_index = 0
    edge_threshold = max(np.mean(portion[max_intensity_row_index, :]) + cfg.edge_threshold_offset , cfg.edge_threshold_min)
    _, binary_mask = cv2.threshold(mag_visual, int(edge_threshold), 255, cv2.THRESH_BINARY)
    noisy_rows = [i for i in range(portion.shape[0]) if np.sum(binary_mask[i] > 0) >= 2]
    noisy_regions = []
    if noisy_rows:
        region_start_var = noisy_rows[0]
        for i in range(1, len(noisy_rows)):
            if noisy_rows[i] - noisy_rows[i-1] > 10:
                noisy_regions.append((region_start_var, noisy_rows[i-1]))
                region_start_var = noisy_rows[i]
        noisy_regions.append((region_start_var, noisy_rows[-1]))

    main_roi_length = end_row - start_row
    filtered_noisy_regions = []
    for current_region_start, current_region_end in noisy_regions:
        # Calculate the overlap with the main ROI (defined by 'start_row' and 'end_row')
        overlap_calc_start = max(current_region_start, start_row)
        overlap_calc_end = min(current_region_end, end_row)
        overlap_length = max(0, overlap_calc_end - overlap_calc_start)
        if overlap_length < 0.5 * main_roi_length:
            filtered_noisy_regions.append((current_region_start, current_region_end))
        else:
            # Updated logging message for clarity about the exclusion reason
            logging.info(f"Excluding region ({current_region_start}, {current_region_end}) because its overlap ({overlap_length:.0f}px) with the main ROI exceeds 50% of ROI length ({0.5 * main_roi_length:.1f}px).")
    
    for start, end in filtered_noisy_regions:
        expanded_start = max(0, start - 2)
        expanded_end = min(len(updated), end + 2)
        if expanded_start >= expanded_end: # Add check for safety
            continue
        before_value = updated[max(0, start - 2)]
        after_value = updated[min(len(updated) - 1, end + 2)]
        transition = np.linspace(before_value, after_value, expanded_end - expanded_start)
        updated[expanded_start:expanded_end] = transition
        
    win = cfg.savgol_window
    win = min(win, len(updated) - 1)
    if win % 2 == 0: win -= 1
    win = max(win, 5)
    smooth = savgol_filter(updated, cfg.savgol_window, cfg.savgol_poly)
    first_deriv = np.gradient(smooth)
    second_deriv = np.gradient(savgol_filter(first_deriv, cfg.savgol_window, cfg.savgol_poly))
    notch = start_row + np.argmin(second_deriv[start_row:end_row])
    cropped_vertical = cropped_horizontal[max(0, notch-ptp//2):min(notch+ptp//2, knee_img.shape[0]), :]
    if return_geometry:
        return cropped_vertical, (int(left_min), int(right_min), int(notch-ptp//2), int(notch+ptp//2))
    return cropped_vertical
