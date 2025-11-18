# --- FINAL DEBUG PYTHON SCRIPT: ID SCANNER V5 (Resolution Fixed) ---

import cv2
import numpy as np
import pytesseract
import imutils
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
import time

# 1. CONFIGURATION
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CAM_INDEX = 0

# --- FRONT COORDINATES (German Layout - Locked In) ---
ROIS_FRONT = {
    "Face": (0.18, 0.82, 0.04, 0.32),
    "Name_Surname": (0.12, 0.35, 0.35, 0.95),
    "Signature": (0.78, 0.95, 0.35, 0.85)
}

# --- 2. MRZ REGION DETECTION ---
def detect_mrz_region_debug(image):
    target_width = 1000
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * scale_ratio)))

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    blackhat_closed = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(blackhat_closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
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


# --- 3. HELPERS ---
def four_point_transform_landscape(image, pts):
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
        dst = np.array([[0, 0], [maxHeight - 1, 0],
                        [maxHeight - 1, maxWidth - 1],
                        [0, maxWidth - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxHeight, maxWidth))
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        dst = np.array([[0, 0], [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return cv2.resize(warped, (1000, 630))


def draw_rois(image, rois_dict, color=(0, 255, 0)):
    h, w = image.shape[:2]
    vis = image.copy()
    for name, (y1, y2, x1, x2) in rois_dict.items():
        cv2.rectangle(vis, (int(w * x1), int(h * y1)),
                      (int(w * x2), int(h * y2)), color, 2)
    return vis


# --- 4. MAIN LOOP ---
collected_results = {}

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise Exception("Camera not found! Check index.")

state = "FRONT"
stabilize_count = 0
REQUIRED_STABLE = 10

print("📸 Scanner Ready. Show FRONT.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, height=600)
        orig = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 15000:
                screenCnt = approx
                break

        display_frame = frame.copy()
        debug_mask_view = None

        if screenCnt is not None:
            cv2.drawContours(display_frame, [screenCnt], -1, (0, 255, 0), 2)
            warped = four_point_transform_landscape(orig, screenCnt.reshape(4, 2))

            # --- FRONT SIDE ---
            if state == "FRONT":
                warped_vis = draw_rois(warped, ROIS_FRONT, color=(0, 255, 255))
                small = cv2.resize(warped_vis, (320, 200))
                display_frame[0:200, 0:320] = small
                cv2.putText(display_frame, "Scan FRONT", (330, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                stabilize_count += 1
                if stabilize_count > REQUIRED_STABLE:
                    h, w = warped.shape[:2]
                    for name, (y1, y2, x1, x2) in ROIS_FRONT.items():
                        collected_results[name] = warped[int(h*y1):int(h*y2),
                                                         int(w*x1):int(w*x2)]
                    state = "WAIT_FLIP"
                    stabilize_count = 0

            # --- WAIT ---
            elif state == "WAIT_FLIP":
                cv2.putText(display_frame, "FLIP TO BACK...", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                stabilize_count += 1
                if stabilize_count > 40:
                    state = "BACK"
                    stabilize_count = 0

            # --- BACK SIDE ---
            elif state == "BACK":
                mrz_box, mrz_roi_img, debug_mask_view = detect_mrz_region_debug(warped)
                warped_vis = cv2.resize(warped, (1000, 630))

                if mrz_box is not None:
                    (mx, my, mw, mh) = mrz_box
                    cv2.rectangle(warped_vis, (mx, my), (mx+mw, my+mh),
                                  (0, 255, 0), 3)
                    cv2.putText(display_frame, "MRZ FOUND!", (330, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    stabilize_count += 1

                    if stabilize_count > 5:
                        gray_roi = cv2.cvtColor(mrz_roi_img, cv2.COLOR_BGR2GRAY)
                        binary = cv2.threshold(gray_roi, 0, 255,
                                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                        text = pytesseract.image_to_string(binary, config=config).strip()

                        if "<<" in text and len(text) > 15:
                            collected_results["MRZ_Image"] = mrz_roi_img
                            collected_results["MRZ_Text"] = text
                            state = "DONE"
                else:
                    cv2.putText(display_frame, "Searching MRZ... (Check X-Ray)",
                                (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    stabilize_count = 0

                small = cv2.resize(warped_vis, (320, 200))
                display_frame[0:200, 0:320] = small

        else:
            stabilize_count = 0

        # Show live windows
        if debug_mask_view is not None:
            cv2.imshow("X-Ray MRZ Mask", debug_mask_view)
        cv2.imshow("ID Scanner", display_frame)

        # Press 'q' to exit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if state == "DONE":
            print("✅ Scanning Complete!")
            time.sleep(1)
            break

except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("="*40)
    print("       🆔 FINAL RESULTS       ")
    print("="*40)

    if collected_results:
        plt.figure(figsize=(15, 8))
        active_keys = [k for k in ["Face", "Name_Surname",
                                   "Signature", "MRZ_Image"]
                       if k in collected_results]

        for i, key in enumerate(active_keys):
            plt.subplot(2, 2, i+1)
            plt.imshow(cv2.cvtColor(collected_results[key],
                                    cv2.COLOR_BGR2RGB))
            plt.title(key)
            plt.axis('off')

        plt.show()

        if "MRZ_Text" in collected_results:
            print("\n📝 MRZ TEXT:\n")
            print(collected_results["MRZ_Text"])
