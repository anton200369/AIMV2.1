"""Robust handheld ID scanner with MRZ extraction.

This script improves stability, lighting tolerance, and output handling so it can
be used reliably in handheld scenarios. The scanner guides the operator through
front and back captures, stabilizes frames before saving regions of interest,
performs MRZ detection/OCR, and persists results for auditing.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import imutils
import numpy as np
import pytesseract

# --------------------------- CONFIGURATION ---------------------------------

ROIS_FRONT: Dict[str, Tuple[float, float, float, float]] = {
    "Face": (0.18, 0.82, 0.04, 0.32),
    "Name_Surname": (0.12, 0.35, 0.35, 0.95),
    "Signature": (0.78, 0.95, 0.35, 0.85),
}


@dataclass
class ScannerConfig:
    cam_index: int = 0
    required_stable: int = 12
    min_contour_area: float = 15000.0
    sharpness_threshold: float = 70.0
    output_dir: Path = Path("captures")
    save_debug: bool = True
    save_warped: bool = True
    tesseract_cmd: Optional[str] = None
    resolution: Tuple[int, int] = (1280, 720)
    show_windows: bool = True
    wait_flip_frames: int = 40
    mrz_stable_frames: int = 5
    roi_preview_size: Tuple[int, int] = (320, 200)
    edge_height: int = 600
    log_prefix: str = "[scanner]"
    timestamps: bool = True
    # Stabilization keeps previous contour to avoid flicker
    _last_contour: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _stabilize_hits: int = field(default=0, init=False, repr=False)


# --------------------------- UTILITIES -------------------------------------

def configure_tesseract(config: ScannerConfig) -> None:
    """Configure pytesseract command, preferring env override."""
    cmd = os.getenv("TESSERACT_CMD", config.tesseract_cmd or "tesseract")
    pytesseract.pytesseract.tesseract_cmd = cmd


def enhance_edges(image: np.ndarray) -> np.ndarray:
    """Return an edge map resilient to handheld capture noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
    )
    v = np.median(adaptive)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(adaptive, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def is_sharp(image: np.ndarray, threshold: float) -> bool:
    return cv2.Laplacian(image, cv2.CV_64F).var() >= threshold


def four_point_transform_landscape(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxHeight > maxWidth:
        dst = np.array(
            [[0, 0], [maxHeight - 1, 0], [maxHeight - 1, maxWidth - 1], [0, maxWidth - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxHeight, maxWidth))
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return cv2.resize(warped, (1000, 630))


def draw_rois(image: np.ndarray, rois_dict: Dict[str, Tuple[float, float, float, float]], color=(0, 255, 0)) -> np.ndarray:
    h, w = image.shape[:2]
    vis = image.copy()
    for name, (y1, y2, x1, x2) in rois_dict.items():
        cv2.rectangle(vis, (int(w * x1), int(h * y1)), (int(w * x2), int(h * y2)), color, 2)
        cv2.putText(vis, name, (int(w * x1) + 5, int(h * y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def detect_document_contour(edges: np.ndarray, config: ScannerConfig) -> Optional[np.ndarray]:
    contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > config.min_contour_area:
            return approx
    return None


def detect_mrz_region_debug(image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], np.ndarray]:
    target_width = 1000
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * scale_ratio)))

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    blackhat_closed = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(blackhat_closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    roi_box = None
    roi_img = None

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])

        if ar > 3 and crWidth > 0.50:
            pad = 10
            p_x = max(0, x - pad)
            p_y = max(0, y - pad)
            p_w = min(image.shape[1], w + (pad * 2))
            p_h = min(image.shape[0], h + (pad * 2))

            roi_img = image[p_y:p_y + p_h, p_x:p_x + p_w].copy()
            roi_box = (p_x, p_y, p_w, p_h)
            break

    return roi_box, roi_img, thresh


def annotate_status(frame: np.ndarray, state: str, message: str) -> None:
    cv2.putText(frame, f"{state}: {message}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


# --------------------------- OCR PIPELINE -----------------------------------

def ocr_mrz(image: np.ndarray) -> str:
    gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    config = r"--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    return pytesseract.image_to_string(binary, config=config).strip()


# --------------------------- MAIN LOOP --------------------------------------

def run_scanner(config: ScannerConfig) -> None:
    configure_tesseract(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(config.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
    if not cap.isOpened():
        raise RuntimeError("Camera not found! Check index or permissions.")

    collected_results: Dict[str, np.ndarray | str] = {}
    state = "FRONT"
    stabilize_count = 0

    print(f"{config.log_prefix} Scanner ready. Show FRONT side.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{config.log_prefix} Unable to read from camera.")
                break

            frame = imutils.resize(frame, height=config.edge_height)
            orig = frame.copy()

            edges = enhance_edges(frame)
            screenCnt = detect_document_contour(edges, config)

            display_frame = frame.copy()
            debug_mask_view = None

            if screenCnt is not None and is_sharp(frame, config.sharpness_threshold):
                cv2.drawContours(display_frame, [screenCnt], -1, (0, 255, 0), 2)
                warped = four_point_transform_landscape(orig, screenCnt.reshape(4, 2))

                if config._last_contour is not None:
                    similarity = cv2.matchShapes(config._last_contour, screenCnt, cv2.CONTOURS_MATCH_I1, 0.0)
                    if similarity < 0.08:
                        stabilize_count += 1
                    else:
                        stabilize_count = 0
                else:
                    stabilize_count += 1
                config._last_contour = screenCnt

                # FRONT SIDE
                if state == "FRONT":
                    warped_vis = draw_rois(warped, ROIS_FRONT, color=(0, 255, 255))
                    small = cv2.resize(warped_vis, config.roi_preview_size)
                    display_frame[0: config.roi_preview_size[1], 0: config.roi_preview_size[0]] = small
                    annotate_status(display_frame, state, "Hold steady - capturing front")

                    if stabilize_count >= config.required_stable:
                        h, w = warped.shape[:2]
                        for name, (y1, y2, x1, x2) in ROIS_FRONT.items():
                            collected_results[name] = warped[int(h * y1): int(h * y2), int(w * x1): int(w * x2)]
                        if config.save_warped:
                            cv2.imwrite(str(config.output_dir / "front_warped.jpg"), warped)
                        state = "WAIT_FLIP"
                        stabilize_count = 0
                        print(f"{config.log_prefix} Front captured. Flip to back.")

                elif state == "WAIT_FLIP":
                    annotate_status(display_frame, state, "Flip card to backside")
                    stabilize_count += 1
                    if stabilize_count > config.wait_flip_frames:
                        state = "BACK"
                        stabilize_count = 0

                elif state == "BACK":
                    mrz_box, mrz_roi_img, debug_mask_view = detect_mrz_region_debug(warped)
                    warped_vis = cv2.resize(warped, (1000, 630))

                    if mrz_box is not None:
                        (mx, my, mw, mh) = mrz_box
                        cv2.rectangle(warped_vis, (mx, my), (mx + mw, my + mh), (0, 255, 0), 3)
                        annotate_status(display_frame, state, "MRZ located - hold steady")
                        stabilize_count += 1

                        if stabilize_count >= config.mrz_stable_frames and mrz_roi_img is not None:
                            text = ocr_mrz(mrz_roi_img)
                            if "<<" in text and len(text) > 15:
                                collected_results["MRZ_Image"] = mrz_roi_img
                                collected_results["MRZ_Text"] = text
                                if config.save_warped:
                                    cv2.imwrite(str(config.output_dir / "back_warped.jpg"), warped)
                                state = "DONE"
                                print(f"{config.log_prefix} MRZ captured and decoded.")
                    else:
                        annotate_status(display_frame, state, "Searching MRZ... adjust position")
                        stabilize_count = 0

                    small = cv2.resize(warped_vis, config.roi_preview_size)
                    display_frame[0: config.roi_preview_size[1], 0: config.roi_preview_size[0]] = small

            else:
                annotate_status(display_frame, state, "Align document inside frame")
                stabilize_count = 0
                config._last_contour = None

            if debug_mask_view is not None and config.show_windows:
                cv2.imshow("X-Ray MRZ Mask", debug_mask_view)
            if config.show_windows:
                cv2.imshow("ID Scanner", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"{config.log_prefix} Stopped by user.")
                break

            if state == "DONE":
                print(f"{config.log_prefix} Scanning complete.")
                time.sleep(1)
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if collected_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S") if config.timestamps else "latest"
            for key, value in collected_results.items():
                if isinstance(value, np.ndarray):
                    filename = f"{timestamp}_{key}.jpg"
                    cv2.imwrite(str(config.output_dir / filename), value)
            if "MRZ_Text" in collected_results:
                txt_path = config.output_dir / f"{timestamp}_mrz.txt"
                txt_path.write_text(collected_results["MRZ_Text"])  # type: ignore[arg-type]

        print("=" * 40)
        print("       🆔 FINAL RESULTS       ")
        print("=" * 40)
        if collected_results:
            for key, val in collected_results.items():
                if isinstance(val, str):
                    print(f"\n📝 {key}:\n{val}")
                else:
                    print(f"✅ Captured {key} -> saved to {config.output_dir}")
        else:
            print("No captures saved. Ensure the document was visible and try again.")


# --------------------------- ENTRYPOINT -------------------------------------

def parse_args() -> ScannerConfig:
    parser = argparse.ArgumentParser(description="Handheld-friendly ID scanner with MRZ decoding.")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index to open")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"), help="Folder to store captures")
    parser.add_argument("--no-windows", action="store_true", help="Disable OpenCV windows (headless)")
    parser.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract executable")
    parser.add_argument("--required-stable", type=int, default=12, help="Frames required for stable capture")
    parser.add_argument("--sharpness", type=float, default=70.0, help="Minimum Laplacian variance")
    parser.add_argument("--min-area", type=float, default=15000.0, help="Minimum contour area for document")
    parser.add_argument("--mrz-stable", type=int, default=5, help="Stable frames before MRZ OCR")
    args = parser.parse_args()

    return ScannerConfig(
        cam_index=args.cam_index,
        output_dir=args.output_dir,
        show_windows=not args.no_windows,
        tesseract_cmd=args.tesseract_cmd,
        required_stable=args.required_stable,
        sharpness_threshold=args.sharpness,
        min_contour_area=args.min_area,
        mrz_stable_frames=args.mrz_stable,
    )


if __name__ == "__main__":
    run_scanner(parse_args())
