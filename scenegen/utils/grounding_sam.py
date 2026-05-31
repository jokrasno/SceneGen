# Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from contextlib import nullcontext


def create_palette():
    # Define a palette with 24 colors for labels 0-23 (example colors)
    palette = [
        0,
        0,
        0,  # Label 0 (black)
        255,
        0,
        0,  # Label 1 (red)
        0,
        255,
        0,  # Label 2 (green)
        0,
        0,
        255,  # Label 3 (blue)
        255,
        255,
        0,  # Label 4 (yellow)
        255,
        0,
        255,  # Label 5 (magenta)
        0,
        255,
        255,  # Label 6 (cyan)
        128,
        0,
        0,  # Label 7 (dark red)
        0,
        128,
        0,  # Label 8 (dark green)
        0,
        0,
        128,  # Label 9 (dark blue)
        128,
        128,
        0,  # Label 10
        128,
        0,
        128,  # Label 11
        0,
        128,
        128,  # Label 12
        64,
        0,
        0,  # Label 13
        0,
        64,
        0,  # Label 14
        0,
        0,
        64,  # Label 15
        64,
        64,
        0,  # Label 16
        64,
        0,
        64,  # Label 17
        0,
        64,
        64,  # Label 18
        192,
        192,
        192,  # Label 19 (light gray)
        128,
        128,
        128,  # Label 20 (gray)
        255,
        165,
        0,  # Label 21 (orange)
        75,
        0,
        130,  # Label 22 (indigo)
        238,
        130,
        238,  # Label 23 (violet)
    ]
    # Extend the palette to have 768 values (256 * 3)
    palette.extend([0] * (768 - len(palette)))
    return palette


PALETTE = create_palette()


# Result Utils
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: Optional[float] = None
    label: Optional[str] = None
    box: Optional[BoundingBox] = None
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


# Utils
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# Post-processing Utils
def generate_colored_segmentation(label_image):
    # Create a PIL Image from the label image (assuming it's a 2D numpy array)
    label_image_pil = Image.fromarray(label_image.astype(np.uint8), mode="P")

    # Apply the palette to the image
    palette = create_palette()
    label_image_pil.putpalette(palette)

    return label_image_pil


def detections_to_label_map(image, detections):
    seg_map = np.zeros(image.size[::-1], dtype=np.uint8)
    for i, detection in enumerate(detections):
        if i >= 254:
            break
        mask = detection.mask
        if mask is None:
            continue
        seg_map[mask > 0] = i + 1
    return Image.fromarray(seg_map, mode="L")


def colorize_label_map(label_map):
    if isinstance(label_map, Image.Image):
        label_array = np.asarray(label_map.convert("L"))
    else:
        label_array = np.asarray(label_map)
    return generate_colored_segmentation(label_array).convert("RGB")


def overlay_label_map(image, label_map, alpha: float = 0.55):
    rgb_array = np.asarray(image.convert("RGB")).copy()
    label_array = np.asarray(label_map.convert("L") if isinstance(label_map, Image.Image) else label_map)
    color_array = np.asarray(colorize_label_map(label_array))
    mask = label_array > 0
    rgb_array[mask] = (
        rgb_array[mask].astype(np.float32) * (1.0 - alpha)
        + color_array[mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return Image.fromarray(rgb_array, mode="RGB")


def plot_segmentation(image, detections):
    label_map = detections_to_label_map(image, detections)
    return colorize_label_map(label_map)


# Grounded SAM
def prepare_model(
    device: str = "cuda",
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
):
    detector_id = (
        detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    )
    object_detector = pipeline(
        model=detector_id, task="zero-shot-object-detection", device=device
    )

    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    processor = AutoProcessor.from_pretrained(segmenter_id)
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)

    return object_detector, processor, segmentator


def detect(
    object_detector: Any,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(
    predictor: Any,
    image: Image.Image,
    boxes: Optional[List[List[List[float]]]] = None,
    point_prompts_by_box: Optional[List[Tuple[List[List[float]], List[int]]]] = None,
    detection_results: Optional[List[Dict[str, Any]]] = None,
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    """
    Use SAM2 predictor to generate masks given an image + a set of bounding boxes.
    """

    if detection_results is None and boxes is None:
        raise ValueError("Either detection_results or detection_boxes must be provided.")

    # Build boxes from detections if not provided
    if boxes is None:
        boxes = get_boxes(detection_results)
    # Flatten legacy [[[...], ...]] input, but keep a single [x0, y0, x1, y1] box intact.
    if (
        isinstance(boxes, list)
        and len(boxes) == 1
        and isinstance(boxes[0], list)
        and len(boxes[0]) > 0
        and isinstance(boxes[0][0], list)
    ):
        boxes = boxes[0]

    # Ensure image is a numpy RGB array (H, W, 3)
    if isinstance(image, Image.Image):
        np_image = np.array(image.convert("RGB"))
    else:
        np_image = np.array(image)

    # Resolve device
    device = getattr(predictor, "device", None)
    if device is None:
        model = getattr(predictor, "model", None)
        if model is not None:
            device = next(model.parameters()).device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare autocast context only for CUDA
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    # Run predictor
    with torch.inference_mode():
        with amp_ctx:
            predictor.set_image(np_image)

            masks_for_boxes = []
            scores_for_boxes = []
            for box_index, box in enumerate(boxes):
                points = None
                point_labels = None
                if point_prompts_by_box is not None and box_index < len(point_prompts_by_box):
                    box_points, box_point_labels = point_prompts_by_box[box_index]
                    if box_points:
                        points = np.asarray(box_points, dtype=np.float32)
                        point_labels = np.asarray(box_point_labels, dtype=np.int32)

                # SAM2ImagePredictor.predict expects original image-space prompts.
                # It handles normalization and model-space transforms internally.
                mask_candidates, score_candidates, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                )
                best_index = int(np.argmax(score_candidates))
                mask = _clip_mask_to_box(mask_candidates[best_index], box, np_image.shape[:2])
                masks_for_boxes.append(mask)
                scores_for_boxes.append(float(score_candidates[best_index]))

            masks = np.stack(masks_for_boxes, axis=0)
            scores = np.asarray(scores_for_boxes, dtype=np.float32)

    # Normalize masks to numpy [N, H, W] boolean
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]  # [N, 1, H, W] -> [N, H, W]
    masks_np = (masks_np > 0).astype(np.uint8)

    # Reuse refine_masks to optionally polygon-refine
    masks_torch = torch.from_numpy(masks_np).unsqueeze(1).to(torch.bool)  # [N,1,H,W]
    masks_list = refine_masks(masks_torch, polygon_refinement)

    if detection_results is None:
        detection_results = [DetectionResult() for _ in masks_list]

    for detection_result, mask in zip(detection_results, masks_list):
        detection_result.mask = mask

    return detection_results


def _clip_mask_to_box(mask: np.ndarray, box: List[float], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return mask
    clipped = np.zeros((height, width), dtype=bool)
    clipped[y0:y1, x0:x1] = True
    return np.asarray(mask).astype(bool) & clipped
