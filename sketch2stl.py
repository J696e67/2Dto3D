#!/usr/bin/env python3
"""
sketch2stl - Convert a 2D sketch/drawing image into a 3D printable STL file.

Usage:
    sketch2stl input.png output.stl [options]
"""

import argparse
import sys
import os
import numpy as np
import cv2
from stl import mesh
import mapbox_earcut


# ---------------------------------------------------------------------------
# Step 1: Image Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    image_path: str,
    blur_radius: int = 5,
    invert: bool = False,
    upsample: int = 2,
) -> np.ndarray:
    """
    Load an image and return a binary (0/255) uint8 array where the sketch
    strokes are WHITE (255) and the background is BLACK (0).

    Parameters
    ----------
    image_path : str
        Path to the input PNG or JPG file.
    blur_radius : int
        Kernel size for Gaussian blur applied after upsampling (must be odd).
    invert : bool
        If True, invert the binarized image before returning.
        Use this when the source image has a dark background.
    upsample : int
        Integer upsampling factor (≥1).  The image is enlarged by this factor
        using Lanczos interpolation before binarization so that findContours
        has more pixels to trace — curves and diagonals become smoother.
        Pixel coordinates are later divided by this factor in the extrusion
        step to preserve real-world mm dimensions.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # Upsample for finer contour resolution
    upsample = max(1, int(upsample))
    if upsample > 1:
        h, w = img.shape
        img = cv2.resize(img, (w * upsample, h * upsample), interpolation=cv2.INTER_LANCZOS4)

    # Ensure blur_radius is odd and at least 1; scale it with upsample
    blur_radius = max(1, blur_radius | 1)
    blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

    # Otsu's binarization — automatically finds the best global threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # After Otsu, dark regions (strokes on white paper) are 0.
    # We want strokes to be WHITE for findContours to detect them as filled
    # regions, so invert.
    binary = cv2.bitwise_not(binary)

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


# ---------------------------------------------------------------------------
# Step 2: Contour / Shape Extraction
# ---------------------------------------------------------------------------

def _smooth_contour(pts: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply circular Gaussian smoothing to the XY coordinates of a closed
    contour, removing pixel-grid staircase noise while preserving shape.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)  float
        Contour vertices.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.
        Values of 1–3 work well for most sketches.

    Returns
    -------
    Smoothed array of the same shape, dtype float64.
    """
    n = len(pts)
    if n < 5 or sigma <= 0:
        return pts.astype(float)

    radius = max(2, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    result = np.empty((n, 2), dtype=float)
    for col in range(2):
        # Wrap-around padding for a closed contour
        padded = np.concatenate([pts[-radius:, col], pts[:, col], pts[:radius, col]])
        # mode='valid' on length-(n + 2*radius) with kernel length-(2*radius+1) → length-n
        result[:, col] = np.convolve(padded.astype(float), kernel, mode="valid")
    return result


def extract_contours(
    binary: np.ndarray,
    min_area: float = 50.0,
    epsilon_factor: float = 0.0005,
    smooth_sigma: float = 1.5,
    use_hierarchy: bool = True,
) -> list[np.ndarray]:
    """
    Find and filter contours from a binary image.

    Parameters
    ----------
    binary : np.ndarray
        Binary image (strokes = 255, background = 0).
    min_area : float
        Contours with area smaller than this are discarded as noise.
    epsilon_factor : float
        Controls how much approxPolyDP simplifies each contour after
        smoothing.  0 = no simplification.  Default is 0.0005 (much
        lower than before) so nearly all curve detail is kept.
    smooth_sigma : float
        Gaussian smoothing sigma applied to raw contour coordinates to
        remove pixel-grid staircase noise.  Set to 0 to disable.
    use_hierarchy : bool
        If True, only keep external (outermost) contours so we don't
        double-extrude nested holes.

    Returns
    -------
    List of contour arrays (each Nx1x2 int32, in pixel space).
    """
    retrieval = cv2.RETR_EXTERNAL if use_hierarchy else cv2.RETR_LIST
    # CHAIN_APPROX_NONE keeps every boundary pixel — no points are lost
    # before our own smoothing/simplification pass.
    contours, _ = cv2.findContours(binary, retrieval, cv2.CHAIN_APPROX_NONE)

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        pts = cnt.reshape(-1, 2).astype(float)

        # 1. Gaussian smooth to remove staircase quantization noise
        pts = _smooth_contour(pts, sigma=smooth_sigma)

        # 2. Optional light simplification (removes near-collinear points)
        if epsilon_factor > 0:
            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * perimeter
            pts_int = pts.round().astype(np.int32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(pts_int, epsilon, True)
            pts = approx.reshape(-1, 2).astype(float)

        if len(pts) >= 3:
            # Store as Nx1x2 int32 for API compatibility; real mm coords
            # are computed later using the float pts stored here.
            filtered.append(pts.reshape(-1, 1, 2).astype(np.float32))

    return filtered


def _process_one_contour(
    cnt: np.ndarray,
    min_area: float,
    epsilon_factor: float,
    smooth_sigma: float,
):
    """Smooth, simplify, and filter a single raw contour.
    Returns Nx1x2 float32 or None if the contour is too small."""
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None
    pts = cnt.reshape(-1, 2).astype(float)
    pts = _smooth_contour(pts, sigma=smooth_sigma)
    if epsilon_factor > 0:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        pts_int = pts.round().astype(np.int32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts_int, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return None
    return pts.reshape(-1, 1, 2).astype(np.float32)


def extract_contour_groups(
    binary: np.ndarray,
    min_area: float = 50.0,
    epsilon_factor: float = 0.0005,
    smooth_sigma: float = 1.5,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Extract contours preserving the parent/hole hierarchy.

    Uses ``RETR_CCOMP`` (two-level hierarchy) so that for each white-stroke
    region both the outer boundary **and** any inner holes are returned.
    This prevents closed outlines (e.g. a heart) from being filled solid —
    inner details (eyes, mouth, etc.) remain visible.

    Returns
    -------
    List of ``(outer_contour, [hole_contours])`` tuples.
    Each contour is an Nx1x2 float32 array in pixel space.
    """
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours or hierarchy is None:
        return []

    hier = hierarchy[0]  # shape (N, 4): [next, prev, child, parent]
    groups: list[tuple[np.ndarray, list[np.ndarray]]] = []

    # Walk top-level (outer) contours — those with parent == -1
    idx = 0
    while idx >= 0:
        if hier[idx][3] != -1:
            idx = hier[idx][0]
            continue

        outer = _process_one_contour(
            contours[idx], min_area, epsilon_factor, smooth_sigma
        )
        if outer is not None:
            holes: list[np.ndarray] = []
            child_idx = hier[idx][2]  # first child (hole)
            while child_idx >= 0:
                hole = _process_one_contour(
                    contours[child_idx], min_area, epsilon_factor, smooth_sigma
                )
                if hole is not None:
                    holes.append(hole)
                child_idx = hier[child_idx][0]  # next sibling hole
            groups.append((outer, holes))

        idx = hier[idx][0]  # next sibling at top level

    return groups


# ---------------------------------------------------------------------------
# Step 3 & 4: 3D Extrusion + STL Generation helpers
# ---------------------------------------------------------------------------

def _ear_clip_triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Triangulate a simple polygon using the ear-clipping algorithm.

    Parameters
    ----------
    polygon : np.ndarray, shape (N, 2)
        Vertices in order.

    Returns
    -------
    List of (i, j, k) index triples into the original polygon array.
    """
    n = len(polygon)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    indices = list(range(n))
    triangles = []

    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_triangle(p, a, b, c):
        d1 = cross2d(p, a, b)
        d2 = cross2d(p, b, c)
        d3 = cross2d(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # Ensure CCW winding so ears have positive cross-product
    signed_area = sum(
        cross2d(polygon[indices[0]], polygon[indices[i]], polygon[indices[i + 1]])
        for i in range(1, len(indices) - 1)
    )
    if signed_area < 0:
        indices.reverse()

    max_iter = n * n + 10  # safety limit
    iters = 0
    while len(indices) > 3 and iters < max_iter:
        iters += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_i = indices[(i - 1) % m]
            curr_i = indices[i]
            next_i = indices[(i + 1) % m]

            a, b, c = polygon[prev_i], polygon[curr_i], polygon[next_i]
            cp = cross2d(a, b, c)
            if cp <= 0:
                continue  # reflex vertex, not an ear

            # Check no other vertex lies inside triangle abc
            is_ear = True
            for j in range(m):
                if indices[j] in (prev_i, curr_i, next_i):
                    continue
                if point_in_triangle(polygon[indices[j]], a, b, c):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((prev_i, curr_i, next_i))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            # Fall back: use fan triangulation for the remainder
            break

    # Remaining indices (3 or fallback fan)
    if len(indices) >= 3:
        for i in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[i], indices[i + 1]))

    return triangles


def _triangulate_with_holes(
    outer: np.ndarray,
    holes: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Triangulate a polygon that may contain holes using mapbox_earcut.

    Parameters
    ----------
    outer : (N, 2) float array — outer ring vertices.
    holes : list of (M, 2) float arrays — hole ring vertices.

    Returns
    -------
    (vertices, triangles)
    vertices : (V, 2) float64 — all ring vertices concatenated.
    triangles : list of (i, j, k) index triples into *vertices*.
    """
    rings = [outer] + holes
    vertices = np.concatenate(rings).astype(np.float64)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)

    tri_flat = mapbox_earcut.triangulate_float64(vertices, ring_ends)
    triangles = [
        (int(tri_flat[i]), int(tri_flat[i + 1]), int(tri_flat[i + 2]))
        for i in range(0, len(tri_flat), 3)
    ]
    return vertices, triangles


def _normal(v0, v1, v2):
    """Compute the unit normal of triangle (v0, v1, v2)."""
    a = v1 - v0
    b = v2 - v0
    n = np.cross(a, b)
    length = np.linalg.norm(n)
    if length == 0:
        return np.array([0.0, 0.0, 1.0])
    return n / length


def _build_base_plate(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_bottom: float, z_top: float,
    margin: float = 2.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return 12 triangles forming a rectangular box (the base plate).
    The box spans [x_min-margin, x_max+margin] x [y_min-margin, y_max+margin]
    x [z_bottom, z_top].
    """
    x0 = x_min - margin
    x1 = x_max + margin
    y0 = y_min - margin
    y1 = y_max + margin
    z0 = z_bottom
    z1 = z_top

    # 8 corners
    corners = {
        "bbl": np.array([x0, y0, z0]),
        "bbr": np.array([x1, y0, z0]),
        "btr": np.array([x1, y1, z0]),
        "btl": np.array([x0, y1, z0]),
        "tbl": np.array([x0, y0, z1]),
        "tbr": np.array([x1, y0, z1]),
        "ttr": np.array([x1, y1, z1]),
        "ttl": np.array([x0, y1, z1]),
    }
    c = corners

    # Each face: two triangles, normals outward
    faces = [
        # Bottom face (normal -Z)
        (c["bbl"], c["btr"], c["bbr"]),
        (c["bbl"], c["btl"], c["btr"]),
        # Top face (normal +Z)
        (c["tbl"], c["tbr"], c["ttr"]),
        (c["tbl"], c["ttr"], c["ttl"]),
        # Front face (-Y)
        (c["bbl"], c["bbr"], c["tbr"]),
        (c["bbl"], c["tbr"], c["tbl"]),
        # Back face (+Y)
        (c["btr"], c["btl"], c["ttl"]),
        (c["btr"], c["ttl"], c["ttr"]),
        # Left face (-X)
        (c["bbl"], c["tbl"], c["ttl"]),
        (c["bbl"], c["ttl"], c["btl"]),
        # Right face (+X)
        (c["bbr"], c["btr"], c["ttr"]),
        (c["bbr"], c["ttr"], c["tbr"]),
    ]
    return faces


def extrude_contours_to_stl(
    contours: list[np.ndarray],
    image_height: int,
    scale: float = 0.1,
    extrude_height: float = 5.0,
    base_thickness: float = 2.0,
    base_margin: float = 2.0,
) -> mesh.Mesh:
    """
    Convert a list of 2D pixel contours into a 3D mesh (numpy-stl Mesh).

    Coordinate system
    -----------------
    * Pixel (col, row) → 3D (col * scale, (img_height - row) * scale, z)
      (Y is flipped so the model is right-side-up when printed.)
    * Z = 0 is the absolute bottom of the base plate.
    * Z = base_thickness is the top of the base / bottom of extrusion.
    * Z = base_thickness + extrude_height is the top of the extrusion.

    Parameters
    ----------
    contours : list of Nx1x2 int32 arrays
        Pixel-space contours (from extract_contours).
    image_height : int
        Height of the source image in pixels (for Y-flip).
    scale : float
        mm per pixel.
    extrude_height : float
        Height of the extruded shape above the base plate (mm).
    base_thickness : float
        Thickness of the base plate (mm).
    base_margin : float
        Extra margin around the sketch for the base plate (mm).
    """
    if not contours:
        raise ValueError("No contours to extrude. The image may be blank or all shapes were filtered out.")

    z_bottom = base_thickness          # bottom of extrusion = top of base
    z_top    = base_thickness + extrude_height

    # Collect all 3D vertices across contours (for bounding box)
    all_x, all_y = [], []

    # Pre-process: convert pixel coords → mm and build per-contour vertex list
    processed = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)  # (N, 2) col, row
        verts_2d = np.column_stack([
            pts[:, 0] * scale,
            (image_height - pts[:, 1]) * scale,
        ])  # mm XY
        all_x.extend(verts_2d[:, 0])
        all_y.extend(verts_2d[:, 1])
        processed.append(verts_2d)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    all_triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- Base plate ---
    base_tris = _build_base_plate(
        x_min, x_max, y_min, y_max,
        z_bottom=0.0, z_top=base_thickness,
        margin=base_margin,
    )
    all_triangles.extend(base_tris)

    # --- Extrude each contour ---
    for verts_2d in processed:
        n = len(verts_2d)
        if n < 3:
            continue

        # 3D top and bottom rings
        top    = np.column_stack([verts_2d, np.full(n, z_top)])
        bottom = np.column_stack([verts_2d, np.full(n, z_bottom)])

        # Side walls: two triangles per edge
        for i in range(n):
            j = (i + 1) % n
            t0, t1 = top[i],    top[j]
            b0, b1 = bottom[i], bottom[j]
            # Outer face: ensure normal points outward
            # We'll rely on winding; verify with _normal later if needed
            all_triangles.append((t0, b0, b1))
            all_triangles.append((t0, b1, t1))

        # Top face triangulation (ear-clip)
        top_tris = _ear_clip_triangulate(verts_2d)
        for (a, b, c) in top_tris:
            v0, v1, v2 = top[a], top[b], top[c]
            # Ensure normal points up (+Z)
            n_vec = _normal(v0, v1, v2)
            if n_vec[2] < 0:
                v0, v2 = v2, v0  # swap to flip normal
            all_triangles.append((v0, v1, v2))

        # Bottom face triangulation (ear-clip, normals should point down)
        bot_tris = _ear_clip_triangulate(verts_2d)
        for (a, b, c) in bot_tris:
            v0, v1, v2 = bottom[a], bottom[b], bottom[c]
            n_vec = _normal(v0, v1, v2)
            if n_vec[2] > 0:
                v0, v2 = v2, v0
            all_triangles.append((v0, v1, v2))

    # Build numpy-stl mesh
    num_tris = len(all_triangles)
    stl_mesh = mesh.Mesh(np.zeros(num_tris, dtype=mesh.Mesh.dtype))
    for i, (v0, v1, v2) in enumerate(all_triangles):
        stl_mesh.vectors[i] = np.array([v0, v1, v2])

    stl_mesh.update_normals()
    return stl_mesh


def _extrude_solid_polygon(
    verts_2d: np.ndarray,
    z_bottom: float,
    z_top: float,
    all_triangles: list,
):
    """Extrude a simple 2D polygon (no holes) between two Z levels.

    Appends side-wall, top-cap, and bottom-cap triangles to *all_triangles*.
    """
    n = len(verts_2d)
    if n < 3:
        return

    top = np.column_stack([verts_2d, np.full(n, z_top)])
    bot = np.column_stack([verts_2d, np.full(n, z_bottom)])

    # Side walls
    for i in range(n):
        j = (i + 1) % n
        all_triangles.append((top[i], bot[i], bot[j]))
        all_triangles.append((top[i], bot[j], top[j]))

    # Top cap (normal +Z)
    for a, b, c in _ear_clip_triangulate(verts_2d):
        v0, v1, v2 = top[a], top[b], top[c]
        if _normal(v0, v1, v2)[2] < 0:
            v0, v2 = v2, v0
        all_triangles.append((v0, v1, v2))

    # Bottom cap (normal -Z)
    for a, b, c in _ear_clip_triangulate(verts_2d):
        v0, v1, v2 = bot[a], bot[b], bot[c]
        if _normal(v0, v1, v2)[2] > 0:
            v0, v2 = v2, v0
        all_triangles.append((v0, v1, v2))


class NotEnclosedError(ValueError):
    """Raised when no closed outline is found for the base shape."""
    pass


def extrude_contour_groups_to_stl(
    contour_groups: list[tuple[np.ndarray, list[np.ndarray]]],
    image_height: int,
    scale: float = 0.1,
    extrude_height: float = 5.0,
    base_thickness: float = 2.0,
    base_margin: float = 2.0,
) -> mesh.Mesh:
    """
    Extrude contour groups (outer + holes) into a watertight 3D STL mesh.

    The base plate follows the shape of the largest enclosed contour
    (the drawn outline that has an interior hole).  If no closed outline
    is found and ``base_thickness > 0``, a ``NotEnclosedError`` is raised
    so the caller can warn the user.

    Parameters
    ----------
    contour_groups : list of (outer, [holes])
        From ``extract_contour_groups``.  Each contour is Nx1x2 float32.
    image_height : int
        Height of the source image in pixels (for Y-flip).
    scale : float
        mm per pixel.
    extrude_height : float
        Height of the extruded shape above the base plate (mm).
    base_thickness : float
        Thickness of the base plate (mm).  Set to 0 to skip the base.
    base_margin : float
        (Reserved for future use with shaped bases.)
    """
    if not contour_groups:
        raise ValueError(
            "No contour groups to extrude. The image may be blank "
            "or all shapes were filtered out."
        )

    # --- Find the base contour: largest group that has holes (= closed outline) ---
    base_group_idx = -1
    max_area = 0.0
    for i, (outer, holes) in enumerate(contour_groups):
        if holes:
            area = cv2.contourArea(outer)
            if area > max_area:
                max_area = area
                base_group_idx = i

    if base_group_idx == -1 and base_thickness > 0:
        raise NotEnclosedError(
            "Your drawing does not have a closed outline. "
            "Please draw a closed boundary around your design "
            "so it can be used as the base shape."
        )

    # --- Convert pixel coords → mm ---
    z_base_bottom = 0.0
    z_base_top = base_thickness
    z_extrude_bottom = base_thickness
    z_extrude_top = base_thickness + extrude_height

    def to_mm(cnt):
        pts = cnt.reshape(-1, 2)
        return np.column_stack([
            pts[:, 0] * scale,
            (image_height - pts[:, 1]) * scale,
        ])

    processed: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for outer, holes in contour_groups:
        outer_2d = to_mm(outer)
        hole_2ds = [to_mm(h) for h in holes]
        processed.append((outer_2d, hole_2ds))

    all_triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- Shaped base plate (follows the largest enclosed contour) ---
    if base_group_idx >= 0 and base_thickness > 0:
        base_2d = processed[base_group_idx][0]
        _extrude_solid_polygon(base_2d, z_base_bottom, z_base_top, all_triangles)

    # --- Extrude each stroke group ---
    for outer_2d, hole_2ds in processed:
        n_outer = len(outer_2d)
        if n_outer < 3:
            continue

        top_outer = np.column_stack([outer_2d, np.full(n_outer, z_extrude_top)])
        bot_outer = np.column_stack([outer_2d, np.full(n_outer, z_extrude_bottom)])

        # Side walls — outer boundary
        for i in range(n_outer):
            j = (i + 1) % n_outer
            all_triangles.append((top_outer[i], bot_outer[i], bot_outer[j]))
            all_triangles.append((top_outer[i], bot_outer[j], top_outer[j]))

        # Side walls — each hole boundary (reversed winding → faces inward)
        for h_2d in hole_2ds:
            nh = len(h_2d)
            if nh < 3:
                continue
            top_h = np.column_stack([h_2d, np.full(nh, z_extrude_top)])
            bot_h = np.column_stack([h_2d, np.full(nh, z_extrude_bottom)])
            for i in range(nh):
                j = (i + 1) % nh
                all_triangles.append((top_h[i], top_h[j], bot_h[j]))
                all_triangles.append((top_h[i], bot_h[j], bot_h[i]))

        # Top & bottom caps — hole-aware triangulation
        if hole_2ds:
            verts_all, tri_idx = _triangulate_with_holes(outer_2d, hole_2ds)
            n_all = len(verts_all)
            top_all = np.column_stack([verts_all, np.full(n_all, z_extrude_top)])
            bot_all = np.column_stack([verts_all, np.full(n_all, z_extrude_bottom)])

            for a, b, c in tri_idx:
                v0, v1, v2 = top_all[a], top_all[b], top_all[c]
                if _normal(v0, v1, v2)[2] < 0:
                    v0, v2 = v2, v0
                all_triangles.append((v0, v1, v2))

            for a, b, c in tri_idx:
                v0, v1, v2 = bot_all[a], bot_all[b], bot_all[c]
                if _normal(v0, v1, v2)[2] > 0:
                    v0, v2 = v2, v0
                all_triangles.append((v0, v1, v2))
        else:
            # No holes — simple ear-clip
            top_tris = _ear_clip_triangulate(outer_2d)
            for a, b, c in top_tris:
                v0, v1, v2 = top_outer[a], top_outer[b], top_outer[c]
                if _normal(v0, v1, v2)[2] < 0:
                    v0, v2 = v2, v0
                all_triangles.append((v0, v1, v2))

            bot_tris = _ear_clip_triangulate(outer_2d)
            for a, b, c in bot_tris:
                v0, v1, v2 = bot_outer[a], bot_outer[b], bot_outer[c]
                if _normal(v0, v1, v2)[2] > 0:
                    v0, v2 = v2, v0
                all_triangles.append((v0, v1, v2))

    # Build numpy-stl mesh
    num_tris = len(all_triangles)
    stl_mesh = mesh.Mesh(np.zeros(num_tris, dtype=mesh.Mesh.dtype))
    for i, (v0, v1, v2) in enumerate(all_triangles):
        stl_mesh.vectors[i] = np.array([v0, v1, v2])

    stl_mesh.update_normals()
    return stl_mesh


# ---------------------------------------------------------------------------
# Visualization helper (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def preview_contours(image: np.ndarray, contours: list, output_path=None):
    """Draw detected contours on the image and show or save a preview."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[preview] matplotlib not installed — skipping preview.", file=sys.stderr)
        return

    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, contours, -1, (0, 200, 0), 2)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Binary Image")
    axes[0].axis("off")
    axes[1].imshow(preview_rgb)
    axes[1].set_title(f"Detected Contours ({len(contours)} shapes)")
    axes[1].axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[preview] Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sketch2stl",
        description="Convert a 2D sketch image (PNG/JPG) into a 3D printable STL file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument("input",  help="Input image file (PNG or JPG).")
    p.add_argument("output", help="Output STL file path.")

    # Preprocessing
    pre = p.add_argument_group("Preprocessing")
    pre.add_argument("--blur",     type=int, default=5,
                     help="Gaussian blur kernel size (odd integer, ≥1).")
    pre.add_argument("--invert",   action="store_true",
                     help="Invert the binarized image (use for dark-background sketches).")
    pre.add_argument("--upsample", type=int, default=2,
                     help="Upsample the image by this integer factor before tracing "
                          "contours. Higher = smoother curves, more triangles. "
                          "Set to 1 to disable.")

    # Contour extraction
    cnt = p.add_argument_group("Contour extraction")
    cnt.add_argument("--min-area",       type=float, default=200.0,
                     help="Minimum contour area in pixels² (of the upsampled image) to keep.")
    cnt.add_argument("--smooth-sigma",   type=float, default=1.5,
                     help="Gaussian smoothing sigma for contour coordinates (pixels). "
                          "Removes pixel-grid staircase noise. 0 = disabled.")
    cnt.add_argument("--epsilon-factor", type=float, default=0.0005,
                     help="Contour simplification factor after smoothing (0 = no simplification).")
    cnt.add_argument("--all-contours",   action="store_true",
                     help="Extract all contours (including nested), not just outermost.")

    # Extrusion
    ext = p.add_argument_group("3D Extrusion")
    ext.add_argument("--scale",          type=float, default=0.1,
                     help="Scale factor: mm per pixel.")
    ext.add_argument("--height",         type=float, default=5.0,
                     help="Extrusion height above the base plate (mm).")
    ext.add_argument("--base-thickness", type=float, default=2.0,
                     help="Base plate thickness (mm).")
    ext.add_argument("--base-margin",    type=float, default=2.0,
                     help="Extra margin around the sketch for the base plate (mm).")

    # Output / debug
    out = p.add_argument_group("Output options")
    out.add_argument("--preview",        metavar="PATH", nargs="?", const="",
                     help="Show a contour preview. Optionally save to PATH (PNG).")
    out.add_argument("--verbose", "-v",  action="store_true",
                     help="Print detailed progress information.")

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    def log(msg):
        if args.verbose:
            print(f"[sketch2stl] {msg}")

    # ------------------------------------------------------------------
    # Step 1: Preprocess
    # ------------------------------------------------------------------
    log(f"Loading and preprocessing: {args.input}")
    try:
        binary = load_and_preprocess(
            args.input,
            blur_radius=args.blur,
            invert=args.invert,
            upsample=args.upsample,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    img_h, img_w = binary.shape
    log(f"Image size after upsampling: {img_w}×{img_h} px")

    # ------------------------------------------------------------------
    # Step 2: Extract contours (hierarchy-aware)
    # ------------------------------------------------------------------
    log("Extracting contours…")
    contour_groups = extract_contour_groups(
        binary,
        min_area=args.min_area,
        epsilon_factor=args.epsilon_factor,
        smooth_sigma=args.smooth_sigma,
    )

    if not contour_groups:
        print(
            "Warning: No contours were found. The image might be blank, or all shapes\n"
            "fell below --min-area. Try --invert, a smaller --min-area, or --verbose.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_contours = sum(1 + len(holes) for _, holes in contour_groups)
    log(f"Found {len(contour_groups)} shape group(s), {total_contours} contour(s) total.")

    if args.verbose:
        for i, (outer, holes) in enumerate(contour_groups):
            area = cv2.contourArea(outer)
            print(f"  Group {i+1}: outer {len(outer)} verts, area={area:.1f} px², "
                  f"{len(holes)} hole(s)")

    # ------------------------------------------------------------------
    # Optional preview (flatten groups for the preview function)
    # ------------------------------------------------------------------
    if args.preview is not None:
        all_cnts = []
        for outer, holes in contour_groups:
            all_cnts.append(outer)
            all_cnts.extend(holes)
        preview_path = args.preview if args.preview else None
        preview_contours(binary, all_cnts, output_path=preview_path)

    # ------------------------------------------------------------------
    # Step 3 & 4: Extrude and write STL
    # ------------------------------------------------------------------
    log("Extruding contours to 3D…")
    effective_scale = args.scale / args.upsample
    try:
        stl_mesh = extrude_contour_groups_to_stl(
            contour_groups,
            image_height=img_h,
            scale=effective_scale,
            extrude_height=args.height,
            base_thickness=args.base_thickness,
            base_margin=args.base_margin,
        )
    except ValueError as e:
        print(f"Error during extrusion: {e}", file=sys.stderr)
        sys.exit(1)

    log(f"Saving STL: {args.output}")
    stl_mesh.save(args.output)

    num_tris = len(stl_mesh.vectors)
    print(
        f"Done! Wrote {args.output}\n"
        f"  Shape groups    : {len(contour_groups)}\n"
        f"  Total contours  : {total_contours}\n"
        f"  Triangles       : {num_tris}\n"
        f"  Extrusion height: {args.height} mm\n"
        f"  Base thickness  : {args.base_thickness} mm\n"
        f"  Scale           : {args.scale} mm/pixel  (upsample ×{args.upsample}, effective {effective_scale:.4f} mm/px)"
    )


if __name__ == "__main__":
    main()
