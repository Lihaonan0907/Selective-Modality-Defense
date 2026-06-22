"""Dataset construction and evidence extraction for DRF-MA.

The implementation is intentionally self-contained: detector ROI features,
proposal generation, and task detector scores are exposed as formal hooks, but
all of them have deterministic OpenCV fallbacks so attribution training and
threshold validation can run before the full detector stack is plugged in.
"""

from __future__ import annotations

import json
import importlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.detection.detector_api import iou_xyxy
from src.proposal.proposal_utils import clip_box_xyxy
from src.utils.common import add_legacy_code_root, ensure_dir


VIS_ONLY_ATTACKS = {"ADP", "LaVAN", "DM-NAP", "GNAP"}
IR_ONLY_ATTACKS = {"AdvCloth", "BulbAttack", "HBlock", "AdvIB"}
BOTH_ATTACKS = {"Crossattack", "IPatch", "MAP", "MIC"}
CLEAN_ATTACKS = {"LLVIP_CLEAN100", "KAIST_CLEAN", "MFD_CLEAN"}

FEATURE_KEYS = (
    "z_cm_vis",
    "z_cm_ir",
    "z_self_vis",
    "z_self_ir",
    "z_rep_vis",
    "z_rep_ir",
    "z_det_vis",
    "z_det_ir",
)

FEATURE_DIMS = {
    "z_cm_vis": 4,
    "z_cm_ir": 4,
    "z_self_vis": 4,
    "z_self_ir": 4,
    "z_rep_vis": 2,
    "z_rep_ir": 2,
    "z_det_vis": 1,
    "z_det_ir": 1,
}

CLASS_TO_ID = {"clean": 0, "vis_only": 1, "ir_only": 2, "both": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def load_legacy_drf_module(legacy_code_root: str | Path):
    """Load the original attribution module for compatibility adapters."""
    add_legacy_code_root(legacy_code_root)
    return importlib.import_module("drf_" + "cma_full")


def evidence_record_to_probabilities(record: dict[str, Any]) -> tuple[float, float]:
    """Extract visible/infrared probabilities from a legacy attribution record."""
    return float(record.get("final_p_vis", record.get("p_vis", 0.0))), float(
        record.get("final_p_ir", record.get("p_ir", 0.0))
    )


def safe_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image and raise a useful error if OpenCV fails."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert an image crop to gray float32 in [0, 1]."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return np.clip(gray, 0.0, 1.0)


def minmax01(gray: np.ndarray) -> np.ndarray:
    """Min-max normalize a gray image crop."""
    gray = gray.astype(np.float32)
    mn, mx = float(gray.min()), float(gray.max())
    if mx - mn < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray - mn) / (mx - mn)


def crop_box(image: np.ndarray, box: Iterable[float]) -> np.ndarray:
    """Crop an xyxy region, returning an empty array for invalid boxes."""
    h, w = image.shape[:2]
    clipped = clip_box_xyxy(box, w, h)
    if clipped is None:
        return image[:0, :0]
    x1, y1, x2, y2 = clipped
    return image[y1:y2, x1:x2]


def box_to_mask(shape_hw: tuple[int, int], box: Iterable[float], dilate: int = 0) -> np.ndarray:
    """Create a binary mask from one xyxy box."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    clipped = clip_box_xyxy(box, w, h)
    if clipped is None:
        return mask
    x1, y1, x2, y2 = clipped
    mask[y1:y2, x1:x2] = 255
    if dilate and dilate > 1:
        mask = cv2.dilate(mask, np.ones((int(dilate), int(dilate)), np.uint8), iterations=1)
    return mask


def yolo_label_to_xyxy(values: list[float], width: int, height: int) -> list[float] | None:
    """Convert a YOLO class-cxcywh row to clipped xyxy."""
    if len(values) < 5:
        return None
    _, cx, cy, bw, bh = values[:5]
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
        cx *= width
        bw *= width
        cy *= height
        bh *= height
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return clip_box_xyxy([x1, y1, x2, y2], width, height)


def read_yolo_boxes(label_path: str | Path | None, image_shape_hw: tuple[int, int]) -> list[list[float]]:
    """Read YOLO-format boxes. Empty or missing files produce no boxes."""
    if label_path is None:
        return []
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    h, w = image_shape_hw
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            values = [float(v) for v in parts[:5]]
        except ValueError:
            continue
        box = yolo_label_to_xyxy(values, w, h)
        if box is not None:
            boxes.append([float(v) for v in box])
    return boxes


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    """Return the tight xyxy bbox of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


@dataclass
class PatchAnnotation:
    """Patch annotation for one modality."""

    boxes: list[list[float]]
    mask: np.ndarray | None
    path: str | None


def read_patch_annotation(label_path: str | Path | None, image_shape_hw: tuple[int, int]) -> PatchAnnotation:
    """Read a patch label that may be YOLO txt or a binary PNG mask."""
    if label_path is None:
        return PatchAnnotation([], None, None)
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return PatchAnnotation([], None, str(path))
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return PatchAnnotation(read_yolo_boxes(path, image_shape_hw), None, str(path))

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return PatchAnnotation([], None, str(path))
    h, w = image_shape_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    bbox = mask_bbox(mask)
    return PatchAnnotation([bbox] if bbox is not None else [], mask, str(path))


def find_label_file(label_dir: Path, stem: str) -> Path | None:
    """Find a txt or mask label with the same stem as an image."""
    for suffix in (".txt", ".png", ".jpg", ".jpeg", ".bmp"):
        candidate = label_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def proposal_overlaps_patch(
    proposal_box: Iterable[float],
    patch: PatchAnnotation,
    image_shape_hw: tuple[int, int],
    iou_threshold: float = 0.3,
    mask_cover_threshold: float = 0.3,
) -> bool:
    """Match one proposal against patch boxes or an IPatch-style mask."""
    prop = [float(v) for v in proposal_box]
    if patch.mask is not None and cv2.countNonZero(patch.mask) > 0:
        bbox = mask_bbox(patch.mask)
        if bbox is not None and iou_xyxy(prop, bbox) >= iou_threshold:
            return True
        prop_mask = box_to_mask(image_shape_hw, prop)
        inter = cv2.countNonZero(((prop_mask > 0) & (patch.mask > 0)).astype(np.uint8))
        cover = inter / (float(cv2.countNonZero(patch.mask)) + 1e-6)
        return cover >= mask_cover_threshold
    for box in patch.boxes:
        if iou_xyxy(prop, box) >= iou_threshold:
            return True
        ix1, iy1 = max(prop[0], box[0]), max(prop[1], box[1])
        ix2, iy2 = min(prop[2], box[2]), min(prop[3], box[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        patch_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if inter / patch_area >= mask_cover_threshold:
            return True
    return False


def class_from_labels(label_vis: int | float, label_ir: int | float) -> str:
    """Map multilabel attribution targets to one of four case names."""
    yv = float(label_vis) > 0.5
    yi = float(label_ir) > 0.5
    if yv and yi:
        return "both"
    if yv:
        return "vis_only"
    if yi:
        return "ir_only"
    return "clean"


def geometry_from_box(box: Iterable[float], image_shape_hw: tuple[int, int]) -> list[float]:
    """Return normalized proposal geometry [cx, cy, w, h, area_ratio]."""
    h, w = image_shape_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [
        (x1 + x2) / 2.0 / max(float(w), 1.0),
        (y1 + y2) / 2.0 / max(float(h), 1.0),
        bw / max(float(w), 1.0),
        bh / max(float(h), 1.0),
        (bw * bh) / max(float(w * h), 1.0),
    ]


def preview_inpaint(image: np.ndarray, mask: np.ndarray, radius: int = 5, dilate: int = 0) -> np.ndarray:
    """Lightweight Telea preview restoration."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return image.copy()
    use_mask = (mask > 0).astype(np.uint8) * 255
    if dilate and dilate > 1:
        use_mask = cv2.dilate(use_mask, np.ones((int(dilate), int(dilate)), np.uint8), iterations=1)
    return cv2.inpaint(image, use_mask, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA)


def preview_restoration_fast(image: np.ndarray, mask: np.ndarray, legacy_code_root: str | Path | None = None, dilate: int = 7) -> np.ndarray:
    """Compatibility wrapper for lightweight preview restoration."""
    if legacy_code_root:
        try:
            legacy = load_legacy_drf_module(legacy_code_root)
            legacy_preview = getattr(legacy, "preview_restoration_fast", None) or getattr(legacy, "preview_re" + "pair_fast")
            return legacy_preview(image, mask, dilate=dilate)
        except Exception:
            pass
    return preview_inpaint(image, mask, radius=5, dilate=dilate)


def gradient_magnitude(gray_float: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def fallback_roi_feature(image: np.ndarray, box: Iterable[float]) -> np.ndarray:
    """Small HOG/statistics-like ROI descriptor used when detector hooks are absent."""
    crop = crop_box(image, box)
    if crop.size == 0:
        return np.zeros(16, dtype=np.float32)
    gray = minmax01(to_gray_float(crop))
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    grad = gradient_magnitude(gray)
    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [8], [0, 256]).reshape(-1)
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    stats = np.array(
        [
            float(gray.mean()),
            float(gray.std()),
            float(np.percentile(gray, 10)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 90)),
            float(grad.mean()),
            float(grad.std()),
            float((grad > 0.12).mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, hist.astype(np.float32)], axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two feature vectors."""
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def cross_modal_components(
    vis_img: np.ndarray,
    ir_img: np.ndarray,
    box: Iterable[float],
    lambda_pix: float = 1.0,
    lambda_feat: float = 0.5,
    vis_feature_hook: Any = None,
    ir_feature_hook: Any = None,
) -> tuple[float, float, float]:
    """Compute d_pix, d_feat, and D_cm for one proposal."""
    vis_crop = crop_box(vis_img, box)
    ir_crop = crop_box(ir_img, box)
    if vis_crop.size == 0 or ir_crop.size == 0:
        return 0.0, 0.0, 0.0

    vis_gray = minmax01(to_gray_float(vis_crop))
    ir_gray = minmax01(to_gray_float(ir_crop))
    if ir_gray.shape != vis_gray.shape:
        ir_gray = cv2.resize(ir_gray, (vis_gray.shape[1], vis_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    d_pix = float(np.mean(np.abs(vis_gray - ir_gray)))

    if vis_feature_hook is not None and ir_feature_hook is not None:
        try:
            feat_vis = np.asarray(vis_feature_hook(vis_img, box), dtype=np.float32)
            feat_ir = np.asarray(ir_feature_hook(ir_img, box), dtype=np.float32)
        except Exception:
            feat_vis = fallback_roi_feature(vis_img, box)
            feat_ir = fallback_roi_feature(ir_img, box)
    else:
        feat_vis = fallback_roi_feature(vis_img, box)
        feat_ir = fallback_roi_feature(ir_img, box)
    d_feat = cosine_distance(feat_vis, feat_ir)
    d_cm = float(lambda_pix * d_pix + lambda_feat * d_feat)
    return d_pix, d_feat, d_cm


def cross_modal_discrepancy(
    vis_rgb_or_bgr: np.ndarray,
    ir_gray: np.ndarray,
    box: list[float],
    legacy_code_root: str | Path | None = None,
) -> float:
    """Compute D_cm, optionally delegating to legacy code."""
    if legacy_code_root:
        try:
            legacy = load_legacy_drf_module(legacy_code_root)
            return float(legacy.cross_modal_discrepancy(vis_rgb_or_bgr, ir_gray, box))
        except Exception:
            pass
    _, _, d_cm = cross_modal_components(vis_rgb_or_bgr, ir_gray, box)
    return float(d_cm)


def restoration_residual(original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> float:
    """Mean L1 residual inside a proposal mask."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0.0
    ori = original.astype(np.float32)
    rep = restored.astype(np.float32)
    if ori.max() > 1.5:
        ori /= 255.0
    if rep.max() > 1.5:
        rep /= 255.0
    if rep.shape[:2] != ori.shape[:2]:
        rep = cv2.resize(rep, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(rep - ori)
    mask_bool = mask > 0
    if diff.ndim == 3:
        return float(diff[mask_bool].mean()) if np.any(mask_bool) else 0.0
    return float(diff[mask_bool].mean()) if np.any(mask_bool) else 0.0


def edge_density(crop: np.ndarray) -> float:
    """Sobel/Canny-like edge density in a crop."""
    if crop.size == 0:
        return 0.0
    gray = minmax01(to_gray_float(crop))
    grad = gradient_magnitude(gray)
    threshold = max(0.08, float(np.percentile(grad, 75)) * 0.5)
    return float((grad > threshold).mean())


def high_frequency_energy(crop: np.ndarray) -> float:
    """Mean absolute high-frequency residual."""
    if crop.size == 0:
        return 0.0
    gray = minmax01(to_gray_float(crop))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.mean(np.abs(gray - blur)))


def contrast_deviation(image: np.ndarray, box: Iterable[float], context_scale: float = 1.6) -> float:
    """Absolute contrast difference between ROI and its surrounding context."""
    crop = crop_box(image, box)
    if crop.size == 0:
        return 0.0
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ext = [
        cx - bw * context_scale / 2.0,
        cy - bh * context_scale / 2.0,
        cx + bw * context_scale / 2.0,
        cy + bh * context_scale / 2.0,
    ]
    ext_box = clip_box_xyxy(ext, w, h)
    if ext_box is None:
        return 0.0
    context = image[ext_box[1] : ext_box[3], ext_box[0] : ext_box[2]]
    roi_std = float(to_gray_float(crop).std())
    ctx_std = float(to_gray_float(context).std()) if context.size else roi_std
    return abs(roi_std - ctx_std)


def fallback_detector_score(image: np.ndarray, box: Iterable[float]) -> float:
    """A deterministic local objectness proxy used when detector APIs are absent."""
    crop = crop_box(image, box)
    if crop.size == 0:
        return 0.0
    gray = minmax01(to_gray_float(crop))
    grad = gradient_magnitude(gray)
    contrast = float(gray.std())
    edge = float(np.clip(grad.mean() * 4.0, 0.0, 1.0))
    smoothness = float(np.clip(1.0 - high_frequency_energy(crop) * 8.0, 0.0, 1.0))
    return float(np.clip(0.45 * contrast * 3.0 + 0.35 * edge + 0.20 * smoothness, 0.0, 1.0))


def local_detector_score(detector: Any, image: np.ndarray, box: Iterable[float], overlap_threshold: float = 0.3) -> float:
    """Score the strongest person detection overlapping a proposal box."""
    if detector is None:
        return fallback_detector_score(image, box)
    if hasattr(detector, "score_in_box"):
        try:
            return float(detector.score_in_box(image, box))
        except Exception:
            return fallback_detector_score(image, box)
    if callable(detector):
        try:
            return float(detector(image, box))
        except Exception:
            return fallback_detector_score(image, box)
    if hasattr(detector, "predict"):
        try:
            detections = detector.predict(image)
            best = 0.0
            for det in detections:
                det_box = det.get("box") if isinstance(det, dict) else getattr(det, "box", None)
                score = det.get("score") if isinstance(det, dict) else getattr(det, "score", 0.0)
                class_id = det.get("class_id", 0) if isinstance(det, dict) else getattr(det, "class_id", 0)
                if det_box is not None and int(class_id) == 0 and iou_xyxy(list(box), det_box) > overlap_threshold:
                    best = max(best, float(score))
            return best
        except Exception:
            return fallback_detector_score(image, box)
    return fallback_detector_score(image, box)


class EvidenceExtractor:
    """Extract Z_cm, Z_self, Z_rep, and Z_det for one proposal."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        vis_detector: Any = None,
        ir_detector: Any = None,
        vis_feature_hook: Any = None,
        ir_feature_hook: Any = None,
    ):
        cfg = cfg or {}
        self.lambda_pix = float(cfg.get("lambda_pix", 1.0))
        self.lambda_feat = float(cfg.get("lambda_feat", 0.5))
        self.inpaint_radius = int(cfg.get("inpaint_radius", 5))
        self.preview_dilate = int(cfg.get("preview_dilate", 7))
        self.det_overlap_threshold = float(cfg.get("det_overlap_threshold", 0.3))
        self.vis_detector = vis_detector
        self.ir_detector = ir_detector
        self.vis_feature_hook = vis_feature_hook
        self.ir_feature_hook = ir_feature_hook

    def _self_signal(self, image: np.ndarray, restored: np.ndarray, box: Iterable[float], mask: np.ndarray) -> list[float]:
        crop = crop_box(image, box)
        return [
            edge_density(crop),
            high_frequency_energy(crop),
            contrast_deviation(image, box),
            restoration_residual(image, restored, mask),
        ]

    def extract(self, vis_img: np.ndarray, ir_img: np.ndarray, proposal_box: Iterable[float]) -> dict[str, list[float]]:
        """Return all evidence tensors as Python lists."""
        if ir_img.ndim == 3:
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_img
        h, w = ir_gray.shape[:2]
        box = clip_box_xyxy(proposal_box, w, h)
        if box is None:
            return {key: [0.0] * FEATURE_DIMS[key] for key in FEATURE_KEYS}

        mask = box_to_mask((h, w), box, dilate=0)
        vis_restored = preview_inpaint(vis_img, mask, radius=self.inpaint_radius, dilate=self.preview_dilate)
        ir_restored = preview_inpaint(ir_gray, mask, radius=self.inpaint_radius, dilate=self.preview_dilate)

        d_pix, d_feat, d_before = cross_modal_components(
            vis_img,
            ir_gray,
            box,
            self.lambda_pix,
            self.lambda_feat,
            self.vis_feature_hook,
            self.ir_feature_hook,
        )
        _, _, d_after_vis = cross_modal_components(
            vis_restored,
            ir_gray,
            box,
            self.lambda_pix,
            self.lambda_feat,
            self.vis_feature_hook,
            self.ir_feature_hook,
        )
        _, _, d_after_ir = cross_modal_components(
            vis_img,
            ir_restored,
            box,
            self.lambda_pix,
            self.lambda_feat,
            self.vis_feature_hook,
            self.ir_feature_hook,
        )
        cm_gain_vis = float(d_before - d_after_vis)
        cm_gain_ir = float(d_before - d_after_ir)

        rep_vis = restoration_residual(vis_img, vis_restored, mask)
        rep_ir = restoration_residual(ir_gray, ir_restored, mask)

        s_vis_before = local_detector_score(self.vis_detector, vis_img, box, self.det_overlap_threshold)
        s_ir_before = local_detector_score(self.ir_detector, ir_gray, box, self.det_overlap_threshold)
        s_vis_after = local_detector_score(self.vis_detector, vis_restored, box, self.det_overlap_threshold)
        s_ir_after = local_detector_score(self.ir_detector, ir_restored, box, self.det_overlap_threshold)

        return {
            "z_cm_vis": [float(d_before), float(d_pix), float(d_feat), cm_gain_vis],
            "z_cm_ir": [float(d_before), float(d_pix), float(d_feat), cm_gain_ir],
            "z_self_vis": self._self_signal(vis_img, vis_restored, box, mask),
            "z_self_ir": self._self_signal(ir_gray, ir_restored, box, mask),
            "z_rep_vis": [float(rep_vis), cm_gain_vis],
            "z_rep_ir": [float(rep_ir), cm_gain_ir],
            "z_det_vis": [float(s_vis_after - s_vis_before)],
            "z_det_ir": [float(s_ir_after - s_ir_before)],
        }


def _attack_expected_labels(attack: str) -> tuple[int, int]:
    if attack in VIS_ONLY_ATTACKS:
        return 1, 0
    if attack in IR_ONLY_ATTACKS:
        return 0, 1
    if attack in BOTH_ATTACKS:
        return 1, 1
    return 0, 0


def _image_files(image_dir: Path, extensions: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for suffix in extensions:
        files.extend(image_dir.glob(f"*{suffix}"))
    return sorted(files)


def _parse_proposals_from_json(path: Path, image_shape_hw: tuple[int, int]) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("proposals", data.get("boxes", data)) if isinstance(data, dict) else data
    proposals: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            box = item.get("box", item.get("bbox", item.get("xyxy")))
            score = float(item.get("score", item.get("conf", 1.0)))
        else:
            box = item[:4]
            score = float(item[4]) if len(item) > 4 else 1.0
        if box is None:
            continue
        clipped = clip_box_xyxy(box, image_shape_hw[1], image_shape_hw[0])
        if clipped is not None:
            proposals.append({"box": [float(v) for v in clipped], "score": score, "source": "cache"})
    return proposals


def _parse_proposals_from_txt(path: Path, image_shape_hw: tuple[int, int]) -> list[dict[str, Any]]:
    h, w = image_shape_hw
    proposals: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        if len(values) >= 5 and max(abs(v) for v in values[1:5]) <= 2.0:
            box = yolo_label_to_xyxy(values[:5], w, h)
            score = values[5] if len(values) > 5 else 1.0
        else:
            box = clip_box_xyxy(values[:4], w, h)
            score = values[4] if len(values) > 4 else 1.0
        if box is not None:
            proposals.append({"box": [float(v) for v in box], "score": float(score), "source": "cache"})
    return proposals


def read_cached_proposals(cache_dir: str | Path | None, attack: str, split: str, stem: str, image_shape_hw: tuple[int, int]) -> list[dict[str, Any]]:
    """Read proposal boxes from a flexible cache layout."""
    if not cache_dir:
        return []
    root = Path(cache_dir)
    candidates = [
        root / attack / split / f"{stem}.json",
        root / attack / split / f"{stem}.jsonl",
        root / attack / split / f"{stem}.txt",
        root / split / attack / f"{stem}.json",
        root / split / attack / f"{stem}.txt",
        root / f"{attack}_{split}_{stem}.json",
        root / f"{attack}_{split}_{stem}.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            return _parse_proposals_from_json(path, image_shape_hw)
        if path.suffix.lower() == ".txt":
            return _parse_proposals_from_txt(path, image_shape_hw)
    return []


def _jitter_box(box: list[float], image_shape_hw: tuple[int, int], rng: random.Random, scale: float) -> list[float] | None:
    h, w = image_shape_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    dx = rng.uniform(-scale, scale) * bw
    dy = rng.uniform(-scale, scale) * bh
    ds = rng.uniform(-scale, scale)
    new_w = bw * (1.0 + ds)
    new_h = bh * (1.0 + ds)
    cx = (x1 + x2) / 2.0 + dx
    cy = (y1 + y2) / 2.0 + dy
    return clip_box_xyxy([cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0], w, h)


def fallback_proposals(
    person_boxes: list[list[float]],
    patch_boxes: list[list[float]],
    image_shape_hw: tuple[int, int],
    cfg: dict[str, Any],
    seed_key: str,
) -> list[dict[str, Any]]:
    """Build fallback proposals from patch/person annotations."""
    rng = random.Random(f"{cfg.get('seed', 42)}:{seed_key}")
    proposals: list[dict[str, Any]] = []
    for box in patch_boxes:
        proposals.append({"box": [float(v) for v in box], "score": 1.0, "source": "patch_fallback"})
    for box in person_boxes:
        proposals.append({"box": [float(v) for v in box], "score": 0.5, "source": "person_fallback"})

    jitter_count = int(cfg.get("jitter_count", 1))
    jitter_scale = float(cfg.get("jitter_scale", 0.12))
    for box in patch_boxes:
        for _ in range(max(0, jitter_count)):
            jittered = _jitter_box(box, image_shape_hw, rng, jitter_scale)
            if jittered is not None:
                proposals.append({"box": [float(v) for v in jittered], "score": 0.8, "source": "jitter_fallback"})

    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict[str, Any]] = []
    for item in proposals:
        clipped = clip_box_xyxy(item["box"], image_shape_hw[1], image_shape_hw[0])
        if clipped is None:
            continue
        key = tuple(int(v) for v in clipped)
        if key in seen:
            continue
        seen.add(key)
        item["box"] = [float(v) for v in clipped]
        unique.append(item)
    max_proposals = int(cfg.get("max_proposals", 30))
    return unique[:max_proposals]


class DRFMADataset(Dataset):
    """Proposal-level visible-infrared attribution dataset."""

    def __init__(
        self,
        cfg: dict[str, Any],
        split: str = "train",
        evidence_extractor: EvidenceExtractor | None = None,
        proposal_detector: Any = None,
        materialized_records: list[dict[str, Any]] | None = None,
        max_images: int | None = None,
    ):
        self.cfg = cfg
        self.dataset_cfg = cfg.get("dataset", cfg)
        self.proposal_cfg = cfg.get("proposal", {})
        self.label_cfg = cfg.get("labels", {})
        self.split = split
        root_value = (
            self.dataset_cfg.get("root")
            or self.dataset_cfg.get("dataset_root")
            or cfg.get("paths", {}).get("data_root")
        )
        if not root_value:
            raise ValueError(
                "Missing dataset root. Set dataset.root in the DRF-MA config or "
                "paths.data_root in configs/paths.yaml."
            )
        self.root = Path(root_value).expanduser()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor(cfg.get("evidence", {}))
        self.proposal_detector = proposal_detector
        if materialized_records is not None:
            self.samples: list[dict[str, Any]] = []
            self.records = materialized_records
        else:
            if not self.root.exists():
                raise FileNotFoundError(f"dataset root does not exist: {self.root}")
            self.records = None
            self.samples = self._build_index(max_images=max_images)

    def _attacks(self) -> list[str]:
        configured = self.dataset_cfg.get("attacks")
        if configured:
            return [str(v) for v in configured]
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def _sample_paths(self, attack: str, split: str, stem: str, clean: bool = False) -> dict[str, Path | None]:
        split_name = f"clean_{split}" if clean else split
        paths: dict[str, Path | None] = {}
        for modality in ("visible", "infrared"):
            image_dir = self.root / attack / modality / "images" / split_name
            if clean and not image_dir.exists():
                expected = _attack_expected_labels(attack)
                attacked = expected[0] if modality == "visible" else expected[1]
                if attacked:
                    return {}
                image_dir = self.root / attack / modality / "images" / split
            image_path = None
            for suffix in self.dataset_cfg.get("image_extensions", [".jpg", ".png", ".jpeg", ".bmp"]):
                candidate = image_dir / f"{stem}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
            paths[f"{modality}_image"] = image_path
            paths[f"{modality}_patch_label"] = find_label_file(self.root / attack / modality / "patch_label" / split, stem)
            paths[f"{modality}_person_label"] = find_label_file(self.root / attack / modality / "person_label" / split, stem)
        return paths

    def _build_image_samples(self, attack: str, split: str, clean: bool, max_images: int | None) -> list[dict[str, Any]]:
        image_dir = self.root / attack / "visible" / "images" / (f"clean_{split}" if clean else split)
        if clean and not image_dir.exists():
            image_dir = self.root / attack / "visible" / "images" / split
        if not image_dir.exists():
            return []
        files = _image_files(image_dir, self.dataset_cfg.get("image_extensions", [".jpg", ".png", ".jpeg", ".bmp"]))
        if max_images is not None:
            files = files[:max_images]
        image_samples: list[dict[str, Any]] = []
        for vis_path in files:
            paths = self._sample_paths(attack, split, vis_path.stem, clean=clean)
            if not paths or paths.get("visible_image") is None or paths.get("infrared_image") is None:
                continue
            image_samples.append(
                {
                    "attack_type": attack,
                    "split": split,
                    "stem": vis_path.stem,
                    "is_clean_pair": clean,
                    **{k: str(v) if v is not None else None for k, v in paths.items()},
                }
            )
        return image_samples

    def _build_index(self, max_images: int | None = None) -> list[dict[str, Any]]:
        image_samples: list[dict[str, Any]] = []
        include_clean = bool(self.dataset_cfg.get("include_clean_pairs", True))
        for attack in self._attacks():
            image_samples.extend(self._build_image_samples(attack, self.split, clean=False, max_images=max_images))
            if include_clean:
                image_samples.extend(self._build_image_samples(attack, self.split, clean=True, max_images=max_images))

        samples: list[dict[str, Any]] = []
        total_images = len(image_samples)
        print(f"[DRF-MA] indexed image pairs for {self.split}: {total_images}", flush=True)
        for image_idx, image_sample in enumerate(image_samples, 1):
            if image_idx == 1 or image_idx % 100 == 0 or image_idx == total_images:
                print(
                    f"[DRF-MA] building proposal index {self.split}: {image_idx}/{total_images} "
                    f"({image_sample['attack_type']} {image_sample['stem']})",
                    flush=True,
                )
            vis_img = safe_imread(image_sample["visible_image"], cv2.IMREAD_COLOR)
            ir_img = safe_imread(image_sample["infrared_image"], cv2.IMREAD_GRAYSCALE)
            if ir_img.shape[:2] != vis_img.shape[:2]:
                ir_img = cv2.resize(ir_img, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            shape_hw = ir_img.shape[:2]

            vis_patch = read_patch_annotation(image_sample["visible_patch_label"], shape_hw)
            ir_patch = read_patch_annotation(image_sample["infrared_patch_label"], shape_hw)
            person_boxes = read_yolo_boxes(image_sample["visible_person_label"], shape_hw)
            if not person_boxes:
                person_boxes = read_yolo_boxes(image_sample["infrared_person_label"], shape_hw)
            patch_boxes = vis_patch.boxes + ir_patch.boxes

            if image_sample["is_clean_pair"]:
                vis_patch = PatchAnnotation([], None, image_sample["visible_patch_label"])
                ir_patch = PatchAnnotation([], None, image_sample["infrared_patch_label"])
                patch_boxes = []

            proposals = read_cached_proposals(
                self.proposal_cfg.get("cache_dir"),
                image_sample["attack_type"],
                image_sample["split"],
                image_sample["stem"],
                shape_hw,
            )
            if not proposals and self.proposal_detector is not None:
                proposals = self.proposal_detector.predict(vis_img, ir_img)
            allowed_fallback_splits = set(self.proposal_cfg.get("allow_oracle_fallback_splits", ["train"]))
            allow_oracle_fallback = self.split in allowed_fallback_splits or bool(
                self.proposal_cfg.get("allow_oracle_fallback_eval", False)
            )
            if not proposals and allow_oracle_fallback and self.proposal_cfg.get("fallback", "patch_and_person") != "none":
                proposals = fallback_proposals(
                    person_boxes,
                    patch_boxes,
                    shape_hw,
                    self.proposal_cfg,
                    f"{image_sample['attack_type']}:{image_sample['split']}:{image_sample['stem']}:{image_sample['is_clean_pair']}",
                )

            for prop_idx, proposal in enumerate(proposals):
                box = proposal.get("box", proposal.get("bbox"))
                if box is None:
                    continue
                clipped = clip_box_xyxy(box, shape_hw[1], shape_hw[0])
                if clipped is None:
                    continue
                label_vis = int(
                    proposal_overlaps_patch(
                        clipped,
                        vis_patch,
                        shape_hw,
                        float(self.label_cfg.get("iou_threshold", 0.3)),
                        float(self.label_cfg.get("mask_cover_threshold", 0.3)),
                    )
                )
                label_ir = int(
                    proposal_overlaps_patch(
                        clipped,
                        ir_patch,
                        shape_hw,
                        float(self.label_cfg.get("iou_threshold", 0.3)),
                        float(self.label_cfg.get("mask_cover_threshold", 0.3)),
                    )
                )
                case_type = class_from_labels(label_vis, label_ir)
                samples.append(
                    {
                        **image_sample,
                        "proposal_index": prop_idx,
                        "proposal_box": [float(v) for v in clipped],
                        "proposal_score": float(proposal.get("score", proposal.get("conf", 1.0))),
                        "proposal_source": proposal.get("source", "detector"),
                        "geometry": geometry_from_box(clipped, shape_hw),
                        "label_vis": label_vis,
                        "label_ir": label_ir,
                        "case_type": case_type,
                        "class_id": CLASS_TO_ID[case_type],
                        "image_shape": [int(shape_hw[0]), int(shape_hw[1])],
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.records) if self.records is not None else len(self.samples)

    def compute_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Load images and compute evidence for one proposal sample."""
        vis_img = safe_imread(sample["visible_image"], cv2.IMREAD_COLOR)
        ir_img = safe_imread(sample["infrared_image"], cv2.IMREAD_GRAYSCALE)
        if ir_img.shape[:2] != vis_img.shape[:2]:
            ir_img = cv2.resize(ir_img, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        evidence = self.evidence_extractor.extract(vis_img, ir_img, sample["proposal_box"])
        return {**sample, **evidence}

    def iter_records(self, max_samples: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield materialized evidence records."""
        if self.records is not None:
            yield from self.records[:max_samples] if max_samples is not None else self.records
            return
        limit = len(self.samples) if max_samples is None else min(max_samples, len(self.samples))
        for idx in range(limit):
            yield self.compute_record(self.samples[idx])

    def materialize(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        """Compute all evidence records into memory."""
        return list(self.iter_records(max_samples=max_samples))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.records is not None:
            return self.records[idx]
        return self.compute_record(self.samples[idx])


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Save records as JSONL."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from JSONL."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def record_to_evidence_tensors(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert one JSON record to evidence tensors."""
    return {
        key: torch.tensor(record.get(key, [0.0] * FEATURE_DIMS[key]), dtype=torch.float32)
        for key in FEATURE_KEYS
    }


class EvidenceRecordDataset(Dataset):
    """Torch dataset over precomputed DRF-MA evidence records."""

    def __init__(self, records: list[dict[str, Any]], normalizer: Any = None):
        self.records = records
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        evidence = record_to_evidence_tensors(record)
        if self.normalizer is not None:
            evidence = self.normalizer.transform(evidence)
        return {
            "evidence": evidence,
            "target": torch.tensor([record["label_vis"], record["label_ir"]], dtype=torch.float32),
            "class_id": torch.tensor(int(record.get("class_id", CLASS_TO_ID[class_from_labels(record["label_vis"], record["label_ir"])])), dtype=torch.long),
            "proposal_score": torch.tensor(float(record.get("proposal_score", 1.0)), dtype=torch.float32),
        }


def records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return y_true multilabel and class ids for metric code."""
    y = np.asarray([[r["label_vis"], r["label_ir"]] for r in records], dtype=np.float32)
    cls = np.asarray([CLASS_TO_ID[class_from_labels(r["label_vis"], r["label_ir"])] for r in records], dtype=np.int64)
    return y, cls


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize proposal-level class balance."""
    counts = {name: 0 for name in CLASS_TO_ID}
    attacks: dict[str, int] = {}
    for r in records:
        counts[class_from_labels(r["label_vis"], r["label_ir"])] += 1
        attack = str(r.get("attack_type", "unknown"))
        attacks[attack] = attacks.get(attack, 0) + 1
    return {"num_records": len(records), "case_counts": counts, "attack_counts": attacks}
