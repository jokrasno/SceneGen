from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from PIL import Image


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


@dataclass
class AutoSegmentationConfig:
    sam2_checkpoint: str = "./checkpoints/sam2-hiera-large/sam2_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2/sam2_hiera_l.yaml"
    device: str = "cuda"
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
        return build_result_from_annotations(rgb_image, annotations, self.config)


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def build_result_from_annotations(
    image: Image.Image,
    annotations: Sequence[dict[str, Any]],
    config: AutoSegmentationConfig | None = None,
) -> AutoSegmentationResult:
    config = config or AutoSegmentationConfig()
    if config.max_instances < 1 or config.max_instances > 254:
        raise ValueError("max_instances must be between 1 and 254 for an L-mode label map")

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    image_area = width * height
    min_area = max(1, int(image_area * config.min_area_ratio))
    max_area = max(1, int(image_area * config.max_area_ratio))

    candidates: list[dict[str, Any]] = []
    for annotation in annotations:
        mask = _annotation_mask(annotation, (height, width))
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        candidates.append(
            {
                "mask": mask,
                "area": area,
                "bbox": bbox,
                "score": _annotation_score(annotation),
                "label": annotation.get("label"),
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

    return AutoSegmentationResult(
        image=rgb_image,
        label_map=Image.fromarray(label_map_array, mode="L"),
        instances=instances,
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
) -> None:
    scene_path = Path(scene_dir)
    scene_path.mkdir(parents=True, exist_ok=True)

    expected_paths = [scene_path / "scene.jpg"]
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
        manifest = {
            "width": result.image.width,
            "height": result.image.height,
            "instances": [
                {
                    "id": instance.id,
                    "file_index": index,
                    "bbox": instance.bbox,
                    "area": instance.area,
                    "score": instance.score,
                    "label": instance.label,
                }
                for index, instance in enumerate(result.instances)
            ],
        }
        (debug_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


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
