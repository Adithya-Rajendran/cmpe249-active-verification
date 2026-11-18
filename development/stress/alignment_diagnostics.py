"""Observe frozen ECC fits and measure two-dimensional plane transfer errors."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def frozen_hashes(root=ROOT):
    manifest = json.loads(re.search(r"```json\n(.*?)\n```", (root / "docs/FREEZE.md").read_text(), re.S)[1])
    return manifest["files"]


def check_frozen(root=ROOT):
    expected = frozen_hashes(root)
    wrong = [name for name, digest in expected.items()
             if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if wrong:
        raise ValueError(f"historical frozen files changed: {wrong}")
    return expected


def load_alignment_only(root=ROOT):
    """Load unchanged pure definitions without importing GPU or newer Python modules."""
    import __future__
    from dataclasses import dataclass
    from typing import Any
    import cv2

    expected = frozen_hashes(root)
    names = {"AlignParams", "_edges", "_pyr_down_mask", "_scaled", "image_align", "_image_align", "_jacobian"}
    module_name = "_stress_frozen_alignment"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace = module.__dict__
    namespace.update(np=np, cv2=cv2, dataclass=dataclass, Any=Any, Box=tuple)
    for name, wanted, dest in [("perception/evidence.py", {"grow"}, {}),
                               ("perception/evidence_v03.py", names, namespace)]:
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f"frozen source mismatch: {name}")
        nodes = [n for n in ast.parse(raw).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
        if {n.name for n in nodes} != wanted:
            raise ValueError(f"missing frozen definitions: {name}")
        dest.update(np=np, Box=tuple)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / name), "exec",
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), dest)
        if wanted == {"grow"}:
            namespace["e021"] = types.SimpleNamespace(grow=dest["grow"])
    return module


def observe_fit(module, ref, query, box, params):
    """Capture H at return without changing the optimizer, guards or returned decision."""
    if sys.gettrace() is not None:
        raise RuntimeError("alignment observation requires an unused trace hook")
    target = module._image_align.__code__
    observed = {}

    def trace(frame, event, arg):
        if frame.f_code is not target:
            return None
        frame.f_trace_lines = False
        if event == "return":
            h = frame.f_locals.get("H")
            if h is not None and np.shape(h) == (3, 3) and np.isfinite(h).all() and abs(h[2, 2]) > 1e-12:
                observed["H"] = np.asarray(h, float).copy() / h[2, 2]
        return trace

    sys.settrace(trace)
    try:
        accepted, info = module.image_align(ref, query, box, params)
    finally:
        sys.settrace(None)
    return accepted, info, observed.get("H")


def project(h, points):
    homogeneous = np.c_[np.asarray(points, float), np.ones(len(points))] @ np.asarray(h, float).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("point projects to infinity")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def point_errors(estimated_query_to_ref, true_query_to_ref, reference_points):
    """Compare query locations of the same reference points, not displacement lengths."""
    estimated = project(np.linalg.inv(estimated_query_to_ref), reference_points)
    actual = project(np.linalg.inv(true_query_to_ref), reference_points)
    residual = estimated - actual
    return {"estimated_query_points": estimated.tolist(), "true_query_points": actual.tolist(),
            "residual_xy_px": residual.tolist(), "errors_px": np.linalg.norm(residual, axis=1).tolist()}


def rendered_to_input_homography(h, scale):
    """Continuous full-size pixels -> resized OpenCV pixel-centre coordinates."""
    transform = np.array([[scale, 0, -0.5], [0, scale, -0.5], [0, 0, 1]], float)
    return transform @ np.asarray(h, float) @ np.linalg.inv(transform)


def plane_diagnostics(raw_h, arrangement, stop, reference_box, input_width):
    from sim.alignment import stop_alignment
    scale = input_width / arrangement["scene"]["cameras"][0]["width"]
    truth = stop_alignment(arrangement, stop, reference_box)
    x0, y0, x1, y1 = np.asarray(reference_box, float) * scale - 0.5
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0+x1)/2, (y0+y1)/2]]
    result = {"direction": "query_to_reference", "coordinates": "OpenCV pixel-centre indices",
              "threshold_px": 5 * input_width / 1200, "planes": {},
              "limitation": "Plane references only. Target residual can include legitimate parallax."}
    if raw_h is None:
        result["fit_available"] = False
        return result
    raw_h = np.asarray(raw_h, dtype=float)
    if raw_h.shape != (3, 3) or not np.isfinite(raw_h).all():
        raise ValueError("diagnostic warp must be a finite 3x3 matrix")
    result["fit_available"] = True
    result["raw_H_query_to_reference"] = raw_h.tolist()
    for name in ("H_front_plane", "H_table_plane", "H_target_plane"):
        if name not in truth:
            continue
        true_h = rendered_to_input_homography(truth[name], scale)
        try:
            values = point_errors(raw_h, true_h, points)
            values["max_error_px"] = max(values["errors_px"])
            values["large_residual"] = values["max_error_px"] > result["threshold_px"]
            result["planes"][name] = values
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["planes"][name] = {"failure": str(exc)}
    return result


def refusal_reason(info):
    text = info.get("refused", "")
    if not text:
        return "accepted"
    if text.startswith("cc "):
        return "correlation"
    if text.startswith("box moves "):
        return "shift"
    if text.startswith("local scale "):
        return "scale"
    return "optimizer_or_input"
