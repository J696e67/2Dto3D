#!/usr/bin/env python3
"""
Flask web UI for sketch2stl — draw or upload a sketch and convert it to STL.

Usage:
    python app.py          # starts on http://127.0.0.1:5000
"""

import json
import os
import base64
import tempfile
from io import BytesIO

import cv2

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS

from sketch2stl import (
    load_and_preprocess,
    load_and_preprocess_color,
    extract_contour_groups,
    extrude_contour_groups_to_stl,
    extrude_multicolor_to_stl,
    NotEnclosedError,
)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/convert", methods=["POST"])
def convert():
    tmp_path = None
    try:
        # --- Receive image (base64 data URL or uploaded file) ---
        if "file" in request.files and request.files["file"].filename:
            uploaded = request.files["file"]
            suffix = os.path.splitext(uploaded.filename)[1] or ".png"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            uploaded.save(tmp_path)
        elif request.form.get("image_data"):
            data_url = request.form["image_data"]
            # Strip "data:image/png;base64," prefix
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
        else:
            return jsonify({"error": "No image provided"}), 400

        # --- Read parameters from form ---
        extrude_height = float(request.form.get("extrude_height", 5.0))
        base_thickness = float(request.form.get("base_thickness", 2.0))
        scale = float(request.form.get("scale", 0.1))
        upsample = int(request.form.get("upsample", 2))
        blur_radius = int(request.form.get("blur_radius", 5))
        min_area = float(request.form.get("min_area", 200.0))
        base_margin = float(request.form.get("base_margin", 2.0))
        invert = request.form.get("invert", "false").lower() == "true"

        # --- Parse optional per-color heights ---
        color_heights_raw = request.form.get("color_heights", "{}")
        color_heights = json.loads(color_heights_raw) if color_heights_raw else {}

        effective_scale = scale / upsample

        if color_heights:
            # --- Multi-color pipeline ---
            # Convert hex keys to RGB tuples and build palette list
            hex_to_rgb = {}
            palette_rgb_list = []
            for hex_color, height in color_heights.items():
                hex_clean = hex_color.lstrip("#")
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
                rgb = (r, g, b)
                hex_to_rgb[hex_color] = rgb
                palette_rgb_list.append(rgb)

            color_masks = load_and_preprocess_color(
                tmp_path,
                palette_colors_rgb=palette_rgb_list,
                blur_radius=blur_radius,
                upsample=upsample,
            )

            # Combine all color masks for base plate determination
            combined_mask = None
            for mask in color_masks.values():
                if combined_mask is None:
                    combined_mask = mask.copy()
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            base_contour_groups = extract_contour_groups(
                combined_mask, min_area=min_area
            ) if combined_mask is not None else []

            # Build per-color contour groups paired with heights
            multi_color_groups = []
            for hex_color, height in color_heights.items():
                rgb = hex_to_rgb[hex_color]
                if rgb not in color_masks:
                    continue
                mask = color_masks[rgb]
                contour_groups = extract_contour_groups(mask, min_area=min_area)
                if contour_groups:
                    multi_color_groups.append((contour_groups, float(height)))

            if not multi_color_groups:
                return jsonify({
                    "error": "No contours found. Try drawing thicker lines, "
                             "lowering min area, or toggling invert."
                }), 422

            image_height = next(iter(color_masks.values())).shape[0]

            stl_mesh = extrude_multicolor_to_stl(
                multi_color_groups,
                image_height=image_height,
                scale=effective_scale,
                base_thickness=base_thickness,
                base_margin=base_margin,
                base_contour_groups=base_contour_groups,
            )
        else:
            # --- Single-color pipeline (backward compat) ---
            binary = load_and_preprocess(
                tmp_path,
                blur_radius=blur_radius,
                invert=invert,
                upsample=upsample,
            )

            contour_groups = extract_contour_groups(binary, min_area=min_area)

            if not contour_groups:
                return jsonify({
                    "error": "No contours found. Try drawing thicker lines, "
                             "lowering min area, or toggling invert."
                }), 422

            stl_mesh = extrude_contour_groups_to_stl(
                contour_groups,
                image_height=binary.shape[0],
                scale=effective_scale,
                extrude_height=extrude_height,
                base_thickness=base_thickness,
                base_margin=base_margin,
            )

        # --- Write STL to memory buffer and return ---
        buf = BytesIO()
        stl_mesh.save("model", fh=buf)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="sketch.stl",
        )

    except NotEnclosedError as e:
        return jsonify({"error": str(e), "type": "not_enclosed"}), 422

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
