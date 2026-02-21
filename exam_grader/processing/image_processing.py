from __future__ import annotations

import cv2
import fitz
import numpy as np
from PIL import Image


def get_page_image_from_pdf(
    pdf_stream: bytes, page_index: int, zoom: float = 6.0
) -> np.ndarray:
    """
    Convert a single PDF page to a high-resolution grayscale numpy array.

    Args:
        pdf_stream: Raw PDF file bytes.
        page_index: Zero-based page index.
        zoom: Scale factor (6.0 ~ 432 DPI from 72 DPI base).

    Returns:
        Grayscale image as numpy array.
    """
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    page = pdf_document.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def get_pdf_page_count(pdf_stream: bytes) -> int:
    """Return the number of pages in a PDF."""
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    return doc.page_count


def get_aligned_pic(
    template: np.ndarray, filled: np.ndarray, scale_factor: float = 1.0
) -> np.ndarray:
    """
    Align a filled form image to a template using SIFT feature matching
    and homography transformation.
    """
    filled_h_orig, filled_w_orig = filled.shape[:2]
    filled_scaled = cv2.resize(
        filled,
        (int(filled_w_orig * scale_factor), int(filled_h_orig * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    filled_h, filled_w = filled_scaled.shape[:2]
    template_resized = cv2.resize(
        template, (filled_w, filled_h), interpolation=cv2.INTER_AREA
    )

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template_resized, None)
    kp2, des2 = sift.detectAndCompute(filled_scaled, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = template_resized.shape[:2]
    aligned = cv2.warpPerspective(filled_scaled, matrix, (w, h))

    return cv2.resize(
        aligned, (filled_w_orig, filled_h_orig), interpolation=cv2.INTER_AREA
    )


def correct_image_rotation(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in a grayscale image using Hough line detection.
    Computes the median angle of near-horizontal lines and rotates to compensate.
    """
    gray = image.copy()
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blur, 150, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles: list[float] = []
    if lines is not None:
        for line in lines:
            _, theta = line[0]
            angle_deg = (theta * 180 / np.pi) - 90
            if abs(angle_deg) < 10:
                angles.append(angle_deg)

    median_angle = float(np.median(angles)) if angles else 0.0
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
    return cv2.warpAffine(
        image, rot_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def draw_white_borders(
    image: np.ndarray, cells: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Draw white border lines around each cell to isolate cell content."""
    border = 10
    for x, y, w, h in cells:
        image[y - border : y + border, x - border : x + w + border] = 255
        image[y + h - border : y + h + border, x - border : x + w + border] = 255
        image[y - border : y + h + border, x - border : x + border] = 255
        image[y - border : y + h + border, x + w - border : x + w + border] = 255
    return image


def detect_symbol(
    image: np.ndarray, cells: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """
    For each cell, find the symbol contour, scale it to fill the cell,
    and center it. Cells without valid symbols are zeroed out.
    """
    for x, y, w, h in cells:
        img_cell = image[y : y + h, x : x + w]
        _, thresh = cv2.threshold(img_cell, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            image[y : y + h, x : x + w] = 0
            continue

        mask = np.zeros_like(thresh)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > 10 and ch > 10:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        bx, by, bw, bh = cv2.boundingRect(mask)

        if bh > 0.4 * h and by < 0.4 * h:
            symbol_crop = img_cell[by : by + bh, bx : bx + bw]
            scale = min(w / bw, h / bh)
            new_w, new_h = int(bw * scale), int(bh * scale)
            resized = cv2.resize(symbol_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            padded = np.zeros((h, w), dtype=np.uint8)
            x_off = (w - new_w) // 2
            y_off = (h - new_h) // 2
            padded[y_off : y_off + new_h, x_off : x_off + new_w] = resized
            image[y : y + h, x : x + w] = padded
        else:
            image[y : y + h, x : x + w] = 0

    return image


def refine_cell_contour(
    aligned_image: np.ndarray,
    cell_bbox: tuple[int, int, int, int],
    margin: int = 10,
) -> tuple[int, int, int, int]:
    """
    Refine a cell bounding box by searching for contours in a padded ROI
    around the template-based coordinates.
    """
    x, y, w, h = cell_bbox
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, aligned_image.shape[1])
    y1 = min(y + h + margin, aligned_image.shape[0])

    roi = aligned_image[y0:y1, x0:x1]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

    _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(roi_thresh, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    roi_thresh = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined_bbox = cell_bbox
    if contours:
        best_contour = None
        best_distance = float("inf")
        expected_center = np.array([w / 2, h / 2])

        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            center = np.array([rx + rw / 2, ry + rh / 2])
            distance = float(np.linalg.norm(center - expected_center))
            if distance < best_distance and 100 < rw * rh < 200:
                best_distance = distance
                best_contour = cnt

        if best_contour is not None:
            rx, ry, rw, rh = cv2.boundingRect(best_contour)
            refined_bbox = (x0 + rx, y0 + ry, rw, rh)

    return refined_bbox


def recalculate_cell(
    aligned_image: np.ndarray,
    template_cell: tuple[int, int, int, int],
    margin: int = 20,
) -> tuple[int, int, int, int]:
    """Refine cell coordinates using contour detection in the aligned image."""
    return refine_cell_contour(aligned_image, template_cell, margin)


def align_image_pipeline(
    image: np.ndarray, template: np.ndarray, scale_factor: float = 0.25
) -> np.ndarray:
    """Full alignment pipeline: match features and warp the image to the template."""
    return get_aligned_pic(template, image, scale_factor)


def style_image(
    aligned_image: np.ndarray, cells: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Process the aligned image for OCR: draw white borders around cells,
    invert to white-on-black, then center symbols within cells.
    """
    draw_white_borders(aligned_image, cells)
    inverted = cv2.bitwise_not(aligned_image)
    return detect_symbol(inverted, cells)
