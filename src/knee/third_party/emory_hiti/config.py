from dataclasses import dataclass
from typing import Tuple
import logging

@dataclass
class CropConfig:
    """
    Hyperparameters for the knee cropping pipeline.
    All defaults match the original script to preserve equivalence.
    """

    # --- Pre-crop / normalization ---
    resize_height: int = 1100  # target height (px), width scaled proportionally
    vertical_trim: int = 50    # pixels trimmed from top and bottom

    # --- Edge detection & outer crop ---
    sobel_threshold: int = 150         # edge magnitude threshold (0–255)
    gaussian_kernel: Tuple[int, int] = (5, 5)  # Gaussian blur kernel size
    sobel_ksize: int = 3               # Sobel kernel size (odd integer)
    line_search_frac: float = 0.1      # fraction of border width/height for line search
    min_vert_line_frac: float = 0.3    # min vertical line length (fraction of image height)
    min_horiz_line_frac: float = 0.3   # minimum horizontal line width (fraction of image width)

    # --- Bilateral split ---
    split_band: Tuple[float, float] = (0.30, 0.60)  # fraction of width to search for separator

    # --- Per-knee cropping ---
    intensity_offset: int = 100        # offset for right knee intensity peak
    clahe_clip: float = 2.0            # CLAHE clip limit
    clahe_tile: Tuple[int, int] = (16, 16)  # CLAHE tile grid
    start_row_frac: float = 0.2        # start of ROI (fraction of height)
    end_row_frac: float = 0.8          # end of ROI (fraction of height)
    black_box_offset: int = 50         # pixel shift away from black box rows
    edge_threshold_offset: int = 5     # added to mean intensity for edge threshold
    edge_threshold_min: int = 100      # minimum edge threshold

    # --- Savitzky–Golay filter ---
    savgol_window: int = 21            # smoothing window (odd integer)
    savgol_poly: int = 3               # polynomial order (< window size)
    
    # --- Morphological operations ---
    morph_kernel: Tuple[int, int] = (5, 5)  # structuring element size for morphological ops

    # --- black_box detection thresholds ---
    black_box_min_area_px: int = 10000

    def __post_init__(self):
        # Clamp fractions
        for name in ["line_search_frac","min_vert_line_frac","min_horiz_line_frac","start_row_frac","end_row_frac"]:
            v = getattr(self, name)
            v2 = max(0.0, min(1.0, float(v)))
            if v2 != v: logging.warning(f"{name} clamped to {v2}")
            setattr(self, name, v2)

        # Ensure split_band valid and ordered
        a, b = self.split_band
        if not (0.0 <= a < b <= 1.0):
            logging.warning(f"split_band invalid {self.split_band}; resetting to (0.30, 0.60)")
            self.split_band = (0.30, 0.60)

        # Ensure odd/positive kernels
        if self.sobel_ksize % 2 == 0 or self.sobel_ksize <= 0:
            nk = max(1, self.sobel_ksize) | 1  # make odd
            logging.warning(f"sobel_ksize corrected to {nk}")
            self.sobel_ksize = nk

        gk = tuple(max(1, k) | 1 for k in self.gaussian_kernel)  # make each odd ≥1
        if gk != self.gaussian_kernel:
            logging.warning(f"gaussian_kernel corrected to {gk}")
            self.gaussian_kernel = gk

        # SavGol: odd window, > poly, ≤ reasonable max
        if self.savgol_window <= self.savgol_poly:
            nw = self.savgol_poly + 2  # make it > poly
            nw = nw | 1                # make odd
            logging.warning(f"savgol_window raised to {nw}")
            self.savgol_window = nw
        if self.savgol_window % 2 == 0:
            self.savgol_window |= 1

        # Non-negatives
        for name in ["resize_height","vertical_trim","intensity_offset","black_box_offset",
                     "edge_threshold_offset","edge_threshold_min","black_box_min_area_px"]:
            v = getattr(self, name)
            if v < 0:
                logging.warning(f"{name} must be ≥0; set to 0")
                setattr(self, name, 0)

        # Tiles/kernels positive
        for name in ["clahe_tile","morph_kernel"]:
            t = tuple(max(1, int(x)) for x in getattr(self, name))
            if t != getattr(self, name):
                logging.warning(f"{name} corrected to {t}")
                setattr(self, name, t)
