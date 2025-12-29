# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 09:23:58 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
FINAL FAST VERSION (CPU optimized)
- Skip grayscale frames entirely
- Detect only strong true-color moving blobs
- Early exit + vectorized NumPy
- Move non-color DCM to Output2
"""

import os
import shutil
import pydicom
import numpy as np
import cv2

# =====================================================
# 路徑設定
# =====================================================
SOURCE_DIR = r"\\SOURCE_DIR "
COLOR_DIR  = r"\\COLOR_DIR"
NON_COLOR_DIR = r"\\NON_COLOR_DIR"
JPG_DIR    = r"\\JPG_DIR"

# =====================================================
# 參數設定（穩定值）
# =====================================================
MOTION_THRESHOLD = 25
MIN_MOVING_PIXELS = 150

ROI_Y1, ROI_Y2 = 0.15, 0.95
ROI_X1, ROI_X2 = 0.15, 0.85

SAT_THRESHOLD = 0.25
DOM_RATIO = 1.35
MIN_COLOR_PIXELS = 200
MIN_INTENSITY = 60

# =====================================================
# 強彩色判斷（向量化）
# =====================================================
def strong_color_mask(rgb):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])

    valid_intensity = max_rgb >= MIN_INTENSITY
    sat = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    valid_sat = sat >= SAT_THRESHOLD

    dom = (
        (r >= DOM_RATIO * np.maximum(g, b)) |
        (g >= DOM_RATIO * np.maximum(r, b)) |
        (b >= DOM_RATIO * np.maximum(r, g))
    )

    return valid_intensity & valid_sat & dom


def frame_has_strong_color_fast(frame_roi):
    return np.count_nonzero(strong_color_mask(frame_roi)) >= MIN_COLOR_PIXELS


def count_strong_color_pixels_fast(pixels):
    r = pixels[:, 0].astype(np.float32)
    g = pixels[:, 1].astype(np.float32)
    b = pixels[:, 2].astype(np.float32)

    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])

    valid_intensity = max_rgb >= MIN_INTENSITY
    sat = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    valid_sat = sat >= SAT_THRESHOLD

    dom = (
        (r >= DOM_RATIO * np.maximum(g, b)) |
        (g >= DOM_RATIO * np.maximum(r, b)) |
        (b >= DOM_RATIO * np.maximum(r, g))
    )

    return np.count_nonzero(valid_intensity & valid_sat & dom)

# =====================================================
# 核心偵測
# =====================================================
def detect_and_save(pixel_data, dcm_name):

    if pixel_data.ndim != 4 or pixel_data.shape[-1] != 3:
        return False

    frame_count, h, w, _ = pixel_data.shape
    y1, y2 = int(h * ROI_Y1), int(h * ROI_Y2)
    x1, x2 = int(w * ROI_X1), int(w * ROI_X2)

    os.makedirs(JPG_DIR, exist_ok=True)

    for i in range(frame_count - 1):

        f1_roi = pixel_data[i, y1:y2, x1:x2]
        f2_roi = pixel_data[i + 1, y1:y2, x1:x2]

        # Step 0: frame-level 彩色快速判斷
        if not frame_has_strong_color_fast(f1_roi):
            continue

        # Step 1: 移動偵測
        diff = np.abs(f1_roi.astype(np.int16) - f2_roi.astype(np.int16))
        motion_mask = np.sum(diff, axis=2) > MOTION_THRESHOLD

        if np.count_nonzero(motion_mask) < MIN_MOVING_PIXELS:
            continue

        motion_u8 = (motion_mask.astype(np.uint8)) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            motion_u8, connectivity=8
        )

        # Step 2: blob level
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < MIN_MOVING_PIXELS:
                continue

            blob_pixels = f1_roi[labels == lab]
            strong_count = count_strong_color_pixels_fast(blob_pixels)
            if strong_count < MIN_COLOR_PIXELS:
                continue

            # Step 3: 存 JPG
            x, y, w_box, h_box, _ = stats[lab]
            crop = f1_roi[y:y+h_box, x:x+w_box]
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            jpg_name = (
                f"{os.path.splitext(dcm_name)[0]}"
                f"_frame{i}_area{area}_color{strong_count}.jpg"
            )
            cv2.imwrite(os.path.join(JPG_DIR, jpg_name), crop_bgr)
            print(f"→ 存 JPG: {jpg_name}")

            return True

    return False

# =====================================================
# 主程式
# =====================================================
def main():
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(NON_COLOR_DIR, exist_ok=True)

    for filename in os.listdir(SOURCE_DIR):
        if not filename.lower().endswith(".dcm"):
            continue

        filepath = os.path.join(SOURCE_DIR, filename)

        try:
            dcm = pydicom.dcmread(filepath)
            pixel_data = dcm.pixel_array
        except Exception as e:
            print(f"讀取錯誤 {filename}: {e}")
            continue

        try:
            if detect_and_save(pixel_data, filename):
                shutil.move(filepath, os.path.join(COLOR_DIR, filename))
                print(f"→ 彩色移動，移到 color/2 : {filename}")
            else:
                shutil.move(filepath, os.path.join(NON_COLOR_DIR, filename))
                print(f"→ 非彩色影像，移到 Output2 : {filename}")
        except Exception as e:
            print(f"處理錯誤 {filename}: {e}")

if __name__ == "__main__":
    main()
    print("over")
