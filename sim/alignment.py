"""True query-to-reference alignment of the simulated views, from the known camera poses. numpy only.

Why: the simulated scenes have no corner ArUco tags any more (shelf_v3, decision 2026-11-03: no fiducials on a home robot's
shelf), so a query view can no longer be registered to its reference through the tags. The simulator knows both camera poses
exactly (the reference at the nominal pose, the query at the arrangement's revisit pose), so the true homography of any plane
between the two views is known. It is EVALUATION-ONLY ground truth: it scores an image-based alignment (evidence v0.3), and is
never an input to the evidence step or to a policy (like view_labels.json and truth.json).

Convention (also written into true_alignment.json):
  - Pixels are continuous coordinates, origin at the top-left corner of the image, u to the right, v down; pixel i covers
    [i, i + 1) (the convention of the annotation boxes and of sim/rig.py). Cameras are the rendered pinhole cameras (no k1).
  - H maps a homogeneous QUERY pixel (u, v, 1) to the REFERENCE pixel: x_ref ~ H x_query. cv2.warpPerspective(query, H, ...) puts
    the query into reference coordinates up to cv2's pixel-centre convention (strictly T(-0.5) H T(0.5); under 0.02 px on
    the v3 revisit poses). H is normalised to H[2, 2] = 1. H_inv maps reference to query.
  - H is exact for the points of one world plane y = y0 (y = the depth behind the shelf's front plane, +y into the shelf).
    Two planes are given: the shelf front plane y = 0 (where the tags used to be) and the vertical plane through the target's
    centre depth (y = the target pose's base y).
  - The same H serves the L1 and L2 queries and all three conditions (visible, occluded, removed), which share the revisit pose.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SCHEMA = "cmpe249.sim-true-alignment.v1"


def camera_matrix(cam: Mapping) -> np.ndarray:
    """3 x 3 intrinsics of a scene-spec camera (square pixels, principal point (W/2, H/2))."""
    f = float(cam["f_px"])
    return np.array([[f, 0.0, cam["width"] / 2], [0.0, f, cam["height"] / 2], [0.0, 0.0, 1.0]])


def plane_matrix(cam: Mapping, y0: float) -> np.ndarray:
    """A (3 x 3) with homogeneous pixel ~ A @ (x, z, 1) for the world points (x, y0, z) of the plane y = y0, for a scene-spec
    camera ({position, right, up, forward, f_px, width, height}; image x along right, image y along -up, depth along forward)."""
    c = np.asarray(cam["position"], float)
    m = np.stack([np.asarray(cam["right"], float), -np.asarray(cam["up"], float), np.asarray(cam["forward"], float)], 1)
    cols = np.stack([np.array([1.0, 0.0, 0.0]) @ m, np.array([0.0, 0.0, 1.0]) @ m, np.array([0.0, y0, 0.0]) @ m - c @ m], 1)
    return camera_matrix(cam) @ cols


def plane_homography(cam_from: Mapping, cam_to: Mapping, y0: float) -> np.ndarray:
    """H with x_to ~ H x_from for the pixels of the plane y = y0 seen by `cam_from` and `cam_to` (normalised, H[2, 2] = 1)."""
    h = plane_matrix(cam_to, y0) @ np.linalg.inv(plane_matrix(cam_from, y0))
    return h / h[2, 2]


def apply(h: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(h, float).T
    return p[:, :2] / p[:, 2:3]


def box_corners(box: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def corner_displacement(h_query_to_ref: np.ndarray, ref_box: Sequence[float]) -> dict:
    """How far the reference box's corners sit from where the query shows them, with no alignment (the identity fallback):
    the corners c of the reference box map into the query at H^-1 c, and the displacement is |H^-1 c - c| per corner (px)."""
    c = box_corners(ref_box)
    d = np.linalg.norm(apply(np.linalg.inv(h_query_to_ref), c) - c, axis=1)
    return {"corners_px": [round(float(v), 3) for v in d], "max_px": round(float(d.max()), 3), "mean_px": round(float(d.mean()), 3)}


def _round(h: np.ndarray) -> list[list[float]]:
    return [[float(f"{v:.15g}") for v in row] for row in h]       # 15 significant digits: the entries span 1e-7 to 1e3


def target_plane_y(arr: Mapping) -> float:
    """Depth (y, m) of the target's centre: the target pose's base y (the body is centred on its base in x and y)."""
    return float(arr["objects"]["target"]["pose"]["base"][1])


def stop_alignment(arr: Mapping, stop: str, ref_box: Sequence[float] | None) -> dict:
    """True query-to-reference alignment of one stop of one arrangement: pose delta, H for the front plane and the target plane,
    and (with the reference box) the corner displacement under each."""
    scene = arr["scene"]
    ref = next(c for c in scene["cameras"] if c["name"] == stop)
    qry = next(c for c in scene["revisit_cameras"] if c["name"] == stop)
    yt = target_plane_y(arr)
    out = {"stop_index": [c["name"] for c in scene["cameras"]].index(stop),
           "pose_delta": arr["revisit"]["stops"][stop]["delta"],
           "H_front_plane": _round(plane_homography(qry, ref, 0.0)),
           "H_target_plane": _round(plane_homography(qry, ref, yt)),
           "reference_box": None if ref_box is None else [int(v) for v in ref_box]}
    if ref_box is not None:
        out["box_corner_displacement_px"] = {
            "front_plane": corner_displacement(np.array(out["H_front_plane"]), ref_box),
            "target_plane": corner_displacement(np.array(out["H_target_plane"]), ref_box)}
    return out


def arrangement_alignment(arr: Mapping, labels: Mapping | None, lighting: str = "L1") -> dict:
    """{target, target_plane_y_m, stops: {stop: stop_alignment}} for one arrangement; the reference box is the one of the
    labels' `lighting` reference mask (L1: the primary pairing's reference)."""
    boxes = None if labels is None else labels["reference_box"][lighting]
    return {"target": arr["target_id"], "target_plane_y_m": round(target_plane_y(arr), 6),
            "stops": {s: stop_alignment(arr, s, None if boxes is None else boxes[s]) for s in arr["stops"]}}


CONVENTION = {
    "H": "3x3 row-major; maps a homogeneous QUERY pixel (u, v, 1) to the REFERENCE pixel (x_ref ~ H x_query), H[2][2] = 1; "
         "cv2.warpPerspective(query, H, (W, H)) is the query in reference coordinates, up to cv2's pixel-centre "
         "convention (strictly T(-0.5) H T(0.5); under 0.02 px on the v3 revisit poses)",
    "pixels": "continuous coordinates, origin at the top-left corner of the image, u right, v down, pixel i covers [i, i + 1) "
              "(the convention of the annotation boxes)",
    "cameras": "the rendered pinhole cameras: reference = scene.cameras (nominal pose), query = scene.revisit_cameras (revisit pose); no lens distortion",
    "planes": {"front_plane": "y = 0 m: the shelf's front plane (where the corner tags were)",
               "target_plane": "y = target_plane_y_m: the vertical plane through the target's centre depth"},
    "exact_for": "world points on the plane only; points off the plane (the rest of the shelf interior) move by parallax",
    "lightings": "the same H serves L1 and L2 queries and the visible, occluded and removed conditions (one revisit pose per stop)",
    "box_corner_displacement_px": "|H^-1 c - c| for the 4 corners c of the reference box (L1 reference mask): how far the box "
                                  "is from where the query shows the target with no alignment (identity fallback)",
}


def doc(items: Sequence[tuple[Mapping, Mapping]], group: str = "random") -> dict:
    """The true_alignment.json document for a list of (arrangement, labels)."""
    return {"schema": SCHEMA, "simulation": True, "evaluation_only": True, "never_an_evidence_or_policy_input": True,
            "group": group,
            "note": "Ground truth from the simulator's camera poses, for scoring an image-based alignment only. Never an input to "
                    "the evidence step or to a policy (like view_labels.json and truth.json).",
            "convention": CONVENTION,
            "arrangements": {a["arrangement_id"]: arrangement_alignment(a, lab) for a, lab in items}}
