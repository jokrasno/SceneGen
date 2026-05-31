from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_AUTO_LABELS = (
    "bed",
    "pillow",
    "chair",
    "desk",
    "table",
    "nightstand",
    "lamp",
    "sofa",
    "curtain",
    "window",
    "artwork",
    "dresser",
    "cabinet",
    "bag",
    "suitcase",
    "door",
    "television",
    "mirror",
)

PERSON_LABELS = {"person", "people", "human", "man", "woman", "child"}
ROOM_SURFACE_LABELS = {"wall", "walls", "floor", "ceiling", "carpet", "rug"}


@dataclass
class DetectionBox:
    bbox: tuple[int, int, int, int]
    label: str | None = None
    score: float | None = None


@dataclass
class InstanceMask:
    id: int
    mask: Image.Image
    bbox: tuple[int, int, int, int]
    area: int
    score: float | None = None
    label: str | None = None


@dataclass
class AutoSegmentationResult:
    image: Image.Image
    label_map: Image.Image
    instances: list[InstanceMask]
    mode: str = "sam2_auto"
    preprocessing: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    quality_passed: bool = True


@dataclass
class AutoSegmentationConfig:
    sam2_checkpoint: str = "./checkpoints/sam2-hiera-large/sam2_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2/sam2_hiera_l.yaml"
    device: str = "cuda"
    segmentation_mode: Literal["hybrid", "manual_boxes", "sam2_auto"] = "hybrid"
    image_preprocess: Literal["auto", "none"] = "auto"
    segment_max_side: int = 2048
    portrait_crop_ratio: float = 1.2
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    min_mask_region_area: int = 100
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.85
    min_new_area_ratio: float = 0.35
    max_instances: int = 16
    sort_by: Literal["score", "area"] = "score"
    detector_id: str = "IDEA-Research/grounding-dino-tiny"
    detection_threshold: float = 0.25
    auto_labels: tuple[str, ...] = DEFAULT_AUTO_LABELS
    exclude_people: bool = True
    include_room_surfaces: bool = False
    allow_low_quality_masks: bool = False
    plane_area_ratio: float = 0.45
    edge_touch_ratio: float = 0.55


class AutoSegmenter:
    def __init__(self, mask_generator: Any, config: AutoSegmentationConfig | None = None):
        self.mask_generator = mask_generator
        self.config = config or AutoSegmentationConfig()

    @classmethod
    def from_sam2(cls, config: AutoSegmentationConfig | None = None) -> "AutoSegmenter":
        config = config or AutoSegmentationConfig()
        checkpoint = Path(config.sam2_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {checkpoint}. "
                "Download it or pass --sam2_checkpoint to use automatic segmentation."
            )

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(
                "SAM2 is required for automatic segmentation. Install the SAM2 package "
                "or run setup with the demo dependencies before using --auto_segment."
            ) from exc

        model = build_sam2(
            config.sam2_model_cfg,
            str(checkpoint),
            device=config.device,
            mode="eval",
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            box_nms_thresh=config.box_nms_thresh,
            crop_n_layers=config.crop_n_layers,
            crop_nms_thresh=config.crop_nms_thresh,
            min_mask_region_area=config.min_mask_region_area,
            output_mode="binary_mask",
        )
        return cls(mask_generator, config)

    def segment(self, image: str | Path | Image.Image) -> AutoSegmentationResult:
        rgb_image = load_rgb_image(image)
        annotations = self.mask_generator.generate(np.asarray(rgb_image))
        return build_result_from_annotations(rgb_image, annotations, self.config, mode="sam2_auto")


class Sam2BoxSegmenter:
    def __init__(self, predictor: Any, config: AutoSegmentationConfig | None = None):
        self.predictor = predictor
        self.config = config or AutoSegmentationConfig()

    @classmethod
    def from_sam2(cls, config: AutoSegmentationConfig | None = None) -> "Sam2BoxSegmenter":
        config = config or AutoSegmentationConfig()
        checkpoint = Path(config.sam2_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {checkpoint}. "
                "Download it or pass --sam2_checkpoint to use box-guided segmentation."
            )

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 is required for box-guided segmentation. Install the SAM2 package "
                "or run setup with the demo dependencies before using hybrid/manual modes."
            ) from exc

        predictor = SAM2ImagePredictor(
            build_sam2(config.sam2_model_cfg, str(checkpoint), device=config.device, mode="eval")
        )
        return cls(predictor, config)

    def segment_boxes(
        self,
        image: str | Path | Image.Image,
        boxes: Sequence[DetectionBox],
        mode: str = "manual_boxes",
        preprocessing: dict[str, Any] | None = None,
        warnings: Sequence[str] | None = None,
    ) -> AutoSegmentationResult:
        rgb_image = load_rgb_image(image)
        if not boxes:
            return build_result_from_annotations(
                rgb_image,
                [],
                self.config,
                mode=mode,
                preprocessing=preprocessing,
                warnings=list(warnings or []) + ["No boxes were available for box-guided segmentation."],
            )

        from scenegen.utils.grounding_sam import BoundingBox, DetectionResult, segment

        detections = [
            DetectionResult(
                score=box.score,
                label=box.label,
                box=BoundingBox(
                    xmin=box.bbox[0],
                    ymin=box.bbox[1],
                    xmax=box.bbox[2],
                    ymax=box.bbox[3],
                ),
            )
            for box in boxes
        ]
        box_list = [list(box.bbox) for box in boxes]
        segmented = segment(
            self.predictor,
            rgb_image,
            boxes=[box_list],
            detection_results=detections,
            polygon_refinement=False,
        )
        annotations = []
        for detection in segmented:
            if detection.mask is None:
                continue
            annotations.append(
                {
                    "segmentation": np.asarray(detection.mask).astype(bool),
                    "score": detection.score,
                    "label": detection.label,
                    "box": detection.box.xyxy if detection.box else None,
                }
            )
        return build_result_from_annotations(
            rgb_image,
            annotations,
            self.config,
            mode=mode,
            preprocessing=preprocessing,
            warnings=warnings,
        )


class ZeroShotBoxDetector:
    def __init__(self, detector: Any, config: AutoSegmentationConfig | None = None):
        self.detector = detector
        self.config = config or AutoSegmentationConfig()

    @classmethod
    def from_transformers(cls, config: AutoSegmentationConfig | None = None) -> "ZeroShotBoxDetector":
        config = config or AutoSegmentationConfig()
        from transformers import pipeline

        detector = pipeline(
            model=config.detector_id,
            task="zero-shot-object-detection",
            device=_transformers_device(config.device),
        )
        return cls(detector, config)

    def detect(self, image: Image.Image) -> list[DetectionBox]:
        candidate_labels = list(self.config.auto_labels)
        if self.config.exclude_people and "person" not in {_clean_label(label) for label in candidate_labels}:
            candidate_labels.append("person")
        labels = [label if label.endswith(".") else label + "." for label in candidate_labels]
        results = self.detector(
            image,
            candidate_labels=labels,
            threshold=self.config.detection_threshold,
        )
        boxes = []
        for result in results:
            raw_box = result.get("box", {})
            label = _clean_label(result.get("label"))
            box = _clip_box(
                (
                    raw_box.get("xmin", 0),
                    raw_box.get("ymin", 0),
                    raw_box.get("xmax", 0),
                    raw_box.get("ymax", 0),
                ),
                image.size,
            )
            if box is None:
                continue
            boxes.append(DetectionBox(bbox=box, label=label, score=result.get("score")))
        return boxes


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def preprocess_image_for_segmentation(
    image: str | Path | Image.Image,
    config: AutoSegmentationConfig | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    config = config or AutoSegmentationConfig()
    raw_image = load_rgb_image(image)
    metadata: dict[str, Any] = {
        "mode": config.image_preprocess,
        "original_size": list(raw_image.size),
        "oriented_size": list(raw_image.size),
        "crop_box": [0, 0, raw_image.width, raw_image.height],
        "scale": 1.0,
        "output_size": list(raw_image.size),
    }

    if config.image_preprocess == "none":
        return raw_image, metadata

    processed = ImageOps.exif_transpose(raw_image).convert("RGB")
    metadata["exif_transposed"] = processed.size != raw_image.size
    metadata["oriented_size"] = list(processed.size)

    width, height = processed.size
    if width > 0 and height / float(width) >= config.portrait_crop_ratio:
        top = max(0, (height - width) // 2)
        bottom = top + width
        crop_box = (0, top, width, bottom)
        processed = processed.crop(crop_box)
        metadata["crop_box"] = list(crop_box)
    else:
        metadata["crop_box"] = [0, 0, width, height]

    max_side = max(processed.size)
    if config.segment_max_side > 0 and max_side > config.segment_max_side:
        scale = config.segment_max_side / float(max_side)
        new_size = (
            max(1, int(round(processed.width * scale))),
            max(1, int(round(processed.height * scale))),
        )
        processed = processed.resize(new_size, Image.Resampling.LANCZOS)
        metadata["scale"] = scale

    metadata["output_size"] = list(processed.size)
    return processed, metadata


def segment_image(
    image: str | Path | Image.Image,
    config: AutoSegmentationConfig | None = None,
    boxes: Sequence[DetectionBox] | None = None,
    detector: ZeroShotBoxDetector | None = None,
    box_segmenter: Sam2BoxSegmenter | None = None,
    auto_segmenter: AutoSegmenter | None = None,
) -> AutoSegmentationResult:
    config = config or AutoSegmentationConfig()
    processed_image, preprocessing = preprocess_image_for_segmentation(image, config)
    mode = config.segmentation_mode

    if mode == "manual_boxes":
        if boxes is None:
            raise ValueError("--boxes_json is required when --segmentation_mode manual_boxes is used")
        transformed_boxes = transform_boxes_for_preprocessed_image(boxes, preprocessing)
        box_segmenter = box_segmenter or Sam2BoxSegmenter.from_sam2(config)
        return box_segmenter.segment_boxes(
            processed_image,
            transformed_boxes,
            mode="manual_boxes",
            preprocessing=preprocessing,
        )

    if mode == "sam2_auto":
        auto_segmenter = auto_segmenter or AutoSegmenter.from_sam2(config)
        result = auto_segmenter.segment(processed_image)
        return _with_preprocessing(result, preprocessing, mode="sam2_auto")

    if mode != "hybrid":
        raise ValueError(f"Unsupported segmentation mode: {mode}")

    warnings: list[str] = []
    try:
        detector = detector or ZeroShotBoxDetector.from_transformers(config)
        detected_boxes = detector.detect(processed_image)
    except Exception as exc:
        detected_boxes = []
        warnings.append(f"Hybrid detector unavailable; falling back to SAM2 auto masks: {exc}")

    if detected_boxes:
        box_segmenter = box_segmenter or Sam2BoxSegmenter.from_sam2(config)
        return box_segmenter.segment_boxes(
            processed_image,
            detected_boxes,
            mode="hybrid",
            preprocessing=preprocessing,
            warnings=warnings,
        )

    warnings.append("Hybrid detector produced no boxes; falling back to SAM2 auto masks.")
    auto_segmenter = auto_segmenter or AutoSegmenter.from_sam2(config)
    result = auto_segmenter.segment(processed_image)
    result = _with_preprocessing(result, preprocessing, mode="sam2_auto_fallback")
    result.warnings = warnings + result.warnings
    return result


def build_result_from_annotations(
    image: Image.Image,
    annotations: Sequence[dict[str, Any]],
    config: AutoSegmentationConfig | None = None,
    mode: str = "sam2_auto",
    preprocessing: dict[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
) -> AutoSegmentationResult:
    config = config or AutoSegmentationConfig()
    if config.max_instances < 1 or config.max_instances > 254:
        raise ValueError("max_instances must be between 1 and 254 for an L-mode label map")

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    image_area = width * height
    min_area = max(1, int(image_area * config.min_area_ratio))
    max_area = max(1, int(image_area * config.max_area_ratio))
    result_warnings = list(warnings or [])
    rejected: list[dict[str, Any]] = []

    entries: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        mask = _annotation_mask(annotation, (height, width))
        area = int(mask.sum())
        bbox = _bbox_from_mask(mask)
        label = _clean_label(annotation.get("label"))
        score = _annotation_score(annotation)
        if bbox is None:
            rejected.append(_rejected_entry(index, label, None, area, "empty_mask", score))
            continue
        entries.append(
            {
                "source_index": index,
                "mask": mask,
                "area": area,
                "bbox": bbox,
                "score": score,
                "label": label,
            }
        )

    excluded_people = np.zeros((height, width), dtype=bool)
    if config.exclude_people:
        for entry in entries:
            if entry["label"] in PERSON_LABELS:
                excluded_people |= entry["mask"]

    candidates: list[dict[str, Any]] = []
    for entry in entries:
        index = entry["source_index"]
        mask = entry["mask"]
        area = entry["area"]
        bbox = entry["bbox"]
        label = entry["label"]
        score = entry["score"]

        if config.exclude_people and label in PERSON_LABELS:
            rejected.append(_rejected_entry(index, label, bbox, area, "excluded_person", score))
            continue

        if excluded_people.any():
            mask = mask & ~excluded_people
            area = int(mask.sum())
            bbox = _bbox_from_mask(mask)
            if bbox is None:
                rejected.append(_rejected_entry(index, label, None, area, "removed_by_excluded_person", score))
                continue

        if area < min_area:
            rejected.append(_rejected_entry(index, label, bbox, area, "too_small", score))
            continue
        if area > max_area:
            rejected.append(_rejected_entry(index, label, bbox, area, "too_large_for_instance", score))
            continue

        reason = _policy_rejection_reason(mask, label, area, image_area, config)
        if reason and not config.allow_low_quality_masks:
            rejected.append(_rejected_entry(index, label, bbox, area, reason, score))
            continue
        if reason:
            result_warnings.append(f"Accepted low-quality mask {index}: {reason}")

        candidates.append(
            {
                "mask": mask,
                "area": area,
                "bbox": bbox,
                "score": score,
                "label": label,
            }
        )

    candidates.sort(key=lambda item: _sort_key(item, config.sort_by), reverse=True)

    label_map_array = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    instances: list[InstanceMask] = []

    for candidate in candidates:
        if len(instances) >= config.max_instances:
            break

        mask = candidate["mask"] & ~occupied
        new_area = int(mask.sum())
        if new_area < min_area:
            continue
        if new_area / float(candidate["area"]) < config.min_new_area_ratio:
            rejected.append(
                _rejected_entry(
                    len(rejected),
                    candidate["label"],
                    candidate["bbox"],
                    new_area,
                    "mostly_overlapped",
                    candidate["score"],
                )
            )
            continue

        instance_id = len(instances) + 1
        label_map_array[mask] = instance_id
        occupied |= mask

        final_bbox = _bbox_from_mask(mask)
        if final_bbox is None:
            continue
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        instances.append(
            InstanceMask(
                id=instance_id,
                mask=mask_image,
                bbox=final_bbox,
                area=new_area,
                score=candidate["score"],
                label=candidate["label"],
            )
        )

    if not instances:
        result_warnings.append("No instance masks passed filtering.")

    return AutoSegmentationResult(
        image=rgb_image,
        label_map=Image.fromarray(label_map_array, mode="L"),
        instances=instances,
        mode=mode,
        preprocessing=preprocessing or {},
        warnings=result_warnings,
        rejected=rejected,
        quality_passed=bool(instances),
    )


def prepare_scene_input(
    image: str | Path | Image.Image,
    scene_dir: str | Path,
    segmenter: AutoSegmenter | None = None,
    config: AutoSegmentationConfig | None = None,
    overwrite: bool = False,
) -> AutoSegmentationResult:
    segmenter = segmenter or AutoSegmenter.from_sam2(config)
    result = segmenter.segment(image)
    save_scene_input(result, scene_dir, overwrite=overwrite)
    return result


def save_scene_input(
    result: AutoSegmentationResult,
    scene_dir: str | Path,
    overwrite: bool = False,
    save_debug: bool = True,
    write_batch_files: bool = True,
) -> None:
    scene_path = Path(scene_dir)
    scene_path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        _clear_generated_scene_input(scene_path)

    expected_paths = [scene_path / "scene.jpg"]
    if write_batch_files:
        for index in range(len(result.instances)):
            expected_paths.extend([scene_path / f"{index}.png", scene_path / f"{index}_mask.png"])
    if not overwrite:
        existing = [path for path in expected_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing auto-segmentation files in {scene_path}: "
                + ", ".join(path.name for path in existing[:5])
            )

    result.image.save(scene_path / "scene.jpg", quality=95)
    rgb_array = np.asarray(result.image)
    combined_mask = np.asarray(result.label_map) > 0
    masked_scene = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
    masked_scene[combined_mask, :3] = rgb_array[combined_mask]
    masked_scene[combined_mask, 3] = 255
    Image.fromarray(masked_scene, mode="RGBA").save(scene_path / "masked_scene.png")

    if write_batch_files:
        for index, instance in enumerate(result.instances):
            mask_array = np.asarray(instance.mask) > 0
            rgba = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
            rgba[mask_array, :3] = rgb_array[mask_array]
            rgba[mask_array, 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(scene_path / f"{index}.png")
            instance.mask.save(scene_path / f"{index}_mask.png")

    if save_debug:
        debug_dir = scene_path / "_auto_segmentation"
        debug_dir.mkdir(exist_ok=True)
        result.label_map.save(debug_dir / "label_map.png")
        _save_label_preview(result.label_map, debug_dir / "label_map_preview.png")
        _save_overlay(result, debug_dir / "segmentation_overlay.png")
        _save_contact_sheet(result, debug_dir / "mask_contact_sheet.png")
        manifest = {
            "mode": result.mode,
            "quality_passed": result.quality_passed,
            "warnings": result.warnings,
            "preprocessing": result.preprocessing,
            "width": result.image.width,
            "height": result.image.height,
            "write_batch_files": write_batch_files,
            "instances": [
                {
                    "id": instance.id,
                    "file_index": index,
                    "bbox": instance.bbox,
                    "area": instance.area,
                    "area_ratio": instance.area / float(result.image.width * result.image.height),
                    "score": instance.score,
                    "label": instance.label,
                }
                for index, instance in enumerate(result.instances)
            ],
            "rejected": result.rejected,
        }
        (debug_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_boxes_json(path: str | Path) -> dict[str | None, list[DetectionBox]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {None: _parse_box_list(data)}
    if isinstance(data, dict) and "boxes" in data:
        return {None: _parse_box_list(data["boxes"])}
    if isinstance(data, dict):
        return {str(key): _parse_box_list(value) for key, value in data.items()}
    raise ValueError("boxes JSON must be a list, an object with 'boxes', or a scene-id mapping")


def transform_boxes_for_preprocessed_image(
    boxes: Sequence[DetectionBox],
    preprocessing: dict[str, Any],
) -> list[DetectionBox]:
    crop = preprocessing.get("crop_box", [0, 0, preprocessing["output_size"][0], preprocessing["output_size"][1]])
    scale = float(preprocessing.get("scale", 1.0))
    output_size = tuple(preprocessing.get("output_size", preprocessing.get("original_size")))
    transformed = []
    for box in boxes:
        left, top, right, bottom = box.bbox
        mapped = (
            int(round((left - crop[0]) * scale)),
            int(round((top - crop[1]) * scale)),
            int(round((right - crop[0]) * scale)),
            int(round((bottom - crop[1]) * scale)),
        )
        clipped = _clip_box(mapped, output_size)
        if clipped is None:
            continue
        transformed.append(DetectionBox(bbox=clipped, label=box.label, score=box.score))
    return transformed


def _parse_box_list(values: Any) -> list[DetectionBox]:
    if not isinstance(values, list):
        raise ValueError("boxes entries must be lists")
    boxes = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("each box must be an object")
        raw_bbox = item.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("each box must provide bbox=[xmin, ymin, xmax, ymax]")
        boxes.append(
            DetectionBox(
                bbox=tuple(int(round(value)) for value in raw_bbox),
                label=_clean_label(item.get("label")),
                score=item.get("score"),
            )
        )
    return boxes


def _annotation_mask(annotation: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        raise ValueError("SAM annotation is missing 'segmentation'")
    if isinstance(segmentation, np.ndarray):
        mask = segmentation
    else:
        raise TypeError(
            "Automatic segmentation expects binary_mask SAM output. "
            "Use output_mode='binary_mask' when constructing the mask generator."
        )
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape}")
    return mask.astype(bool)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _annotation_score(annotation: dict[str, Any]) -> float | None:
    values = [
        annotation.get("predicted_iou"),
        annotation.get("stability_score"),
        annotation.get("score"),
    ]
    scores = [float(value) for value in values if value is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _sort_key(item: dict[str, Any], sort_by: str) -> tuple[float, int]:
    score = item["score"]
    score_value = float(score) if score is not None else -1.0
    area = int(item["area"])
    if sort_by == "area":
        return float(area), area
    return score_value, area


def _policy_rejection_reason(
    mask: np.ndarray,
    label: str | None,
    area: int,
    image_area: int,
    config: AutoSegmentationConfig,
) -> str | None:
    label_key = _clean_label(label)
    if config.exclude_people and label_key in PERSON_LABELS:
        return "excluded_person"
    if not config.include_room_surfaces and label_key in ROOM_SURFACE_LABELS:
        return "excluded_room_surface"

    area_ratio = area / float(image_area)
    if area_ratio > config.plane_area_ratio:
        return "large_plane_like_mask"
    if area_ratio > 0.10 and _edge_touch_ratio(mask) >= config.edge_touch_ratio:
        return "edge_touching_plane_like_mask"
    return None


def _edge_touch_ratio(mask: np.ndarray) -> float:
    height, width = mask.shape
    if height == 0 or width == 0:
        return 0.0
    perimeter = (2 * width) + (2 * height)
    edge_pixels = (
        int(mask[0, :].sum())
        + int(mask[-1, :].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
    )
    return edge_pixels / float(perimeter)


def _rejected_entry(
    index: int,
    label: str | None,
    bbox: tuple[int, int, int, int] | None,
    area: int,
    reason: str,
    score: float | None,
) -> dict[str, Any]:
    return {
        "source_index": index,
        "label": label,
        "bbox": bbox,
        "area": area,
        "reason": reason,
        "score": score,
    }


def _clean_label(label: Any) -> str | None:
    if label is None:
        return None
    value = str(label).strip().lower()
    if value.endswith("."):
        value = value[:-1]
    return value or None


def _clip_box(
    bbox: Sequence[int | float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    left = max(0, min(width, int(round(float(bbox[0])))))
    top = max(0, min(height, int(round(float(bbox[1])))))
    right = max(0, min(width, int(round(float(bbox[2])))))
    bottom = max(0, min(height, int(round(float(bbox[3])))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _transformers_device(device: str) -> int | str:
    if device == "cpu":
        return -1
    if device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return device


def _with_preprocessing(
    result: AutoSegmentationResult,
    preprocessing: dict[str, Any],
    mode: str,
) -> AutoSegmentationResult:
    result.preprocessing = preprocessing
    result.mode = mode
    return result


def _save_label_preview(label_map: Image.Image, path: Path) -> None:
    labels = np.asarray(label_map.convert("L"))
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = _preview_palette()
    for label in np.unique(labels):
        if label == 0:
            continue
        colors[labels == label] = palette[int(label) % len(palette)]
    Image.fromarray(colors, mode="RGB").save(path)


def _save_overlay(result: AutoSegmentationResult, path: Path) -> None:
    base = np.asarray(result.image).astype(np.float32)
    labels = np.asarray(result.label_map.convert("L"))
    overlay = base.copy()
    palette = _preview_palette()
    for label in np.unique(labels):
        if label == 0:
            continue
        color = np.asarray(palette[int(label) % len(palette)], dtype=np.float32)
        mask = labels == label
        overlay[mask] = (overlay[mask] * 0.55) + (color * 0.45)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _save_contact_sheet(result: AutoSegmentationResult, path: Path) -> None:
    thumb_size = (240, 240)
    if not result.instances:
        Image.new("RGB", thumb_size, "black").save(path)
        return

    cols = min(4, len(result.instances))
    rows = int(np.ceil(len(result.instances) / float(cols)))
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "black")
    draw = ImageDraw.Draw(sheet)
    rgb = np.asarray(result.image)
    for index, instance in enumerate(result.instances):
        mask = np.asarray(instance.mask.convert("L")) > 0
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask, :3] = rgb[mask]
        rgba[mask, 3] = 255
        tile = Image.fromarray(rgba, mode="RGBA")
        bbox = instance.bbox
        tile = tile.crop(bbox)
        tile.thumbnail((thumb_size[0], thumb_size[1] - 24), Image.Resampling.LANCZOS)
        background = Image.new("RGB", thumb_size, "black")
        x = (thumb_size[0] - tile.width) // 2
        y = 20 + ((thumb_size[1] - 24 - tile.height) // 2)
        background.paste(tile.convert("RGB"), (x, y), tile)
        col = index % cols
        row = index // cols
        sheet.paste(background, (col * thumb_size[0], row * thumb_size[1]))
        label = instance.label or f"mask {instance.id}"
        draw.text((col * thumb_size[0] + 8, row * thumb_size[1] + 6), label[:28], fill="white")
    sheet.save(path)


def _preview_palette() -> list[tuple[int, int, int]]:
    return [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
    ]


def _clear_generated_scene_input(scene_path: Path) -> None:
    for path in scene_path.iterdir() if scene_path.exists() else []:
        if path.is_dir() and path.name == "_auto_segmentation":
            shutil.rmtree(path)
            continue
        if not path.is_file():
            continue
        if path.name in {"scene.jpg", "masked_scene.png"}:
            path.unlink()
        elif path.name.endswith("_mask.png"):
            path.unlink()
        elif path.suffix == ".png" and path.stem.isdigit():
            path.unlink()
