import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scenegen.segmentation import (
    AutoSegmentationConfig,
    DetectionBox,
    build_result_from_annotations,
    preprocess_image_for_segmentation,
    save_scene_input,
    segment_image,
    transform_boxes_for_preprocessed_image,
)


class AutoSegmentTest(unittest.TestCase):
    def test_builds_l_mode_instance_label_map(self):
        image = Image.new("RGB", (20, 20), "white")
        large_mask = np.zeros((20, 20), dtype=bool)
        large_mask[1:10, 1:10] = True
        small_overlap = np.zeros((20, 20), dtype=bool)
        small_overlap[4:12, 4:12] = True

        result = build_result_from_annotations(
            image,
            [
                {"segmentation": large_mask, "predicted_iou": 0.95, "stability_score": 0.95},
                {"segmentation": small_overlap, "predicted_iou": 0.90, "stability_score": 0.90},
            ],
            AutoSegmentationConfig(
                min_area_ratio=0.01,
                max_area_ratio=0.9,
                min_new_area_ratio=0.25,
            ),
        )

        self.assertEqual(result.label_map.mode, "L")
        self.assertEqual(len(result.instances), 2)
        labels = set(np.unique(np.asarray(result.label_map)).tolist())
        self.assertEqual(labels, {0, 1, 2})

    def test_save_scene_input_uses_batch_inference_layout(self):
        image = Image.new("RGB", (10, 10), "white")
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        result = build_result_from_annotations(
            image,
            [{"segmentation": mask, "predicted_iou": 0.9, "stability_score": 0.9}],
            AutoSegmentationConfig(min_area_ratio=0.01),
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_scene_input(result, tmp)
            names = {path.name for path in Path(tmp).iterdir()}

        self.assertIn("scene.jpg", names)
        self.assertIn("0.png", names)
        self.assertIn("0_mask.png", names)
        self.assertIn("masked_scene.png", names)
        self.assertIn("_auto_segmentation", names)
        self.assertNotIn("label_map.png", names)

    def test_preprocess_auto_crops_portrait_and_scales(self):
        image = Image.new("RGB", (100, 160), "white")

        processed, metadata = preprocess_image_for_segmentation(
            image,
            AutoSegmentationConfig(
                image_preprocess="auto",
                segment_max_side=50,
                portrait_crop_ratio=1.2,
            ),
        )

        self.assertEqual(processed.size, (50, 50))
        self.assertEqual(metadata["crop_box"], [0, 30, 100, 130])
        self.assertEqual(metadata["output_size"], [50, 50])
        self.assertAlmostEqual(metadata["scale"], 0.5)

    def test_manual_boxes_transform_through_preprocessing(self):
        _, metadata = preprocess_image_for_segmentation(
            Image.new("RGB", (100, 160), "white"),
            AutoSegmentationConfig(
                image_preprocess="auto",
                segment_max_side=50,
                portrait_crop_ratio=1.2,
            ),
        )

        boxes = transform_boxes_for_preprocessed_image(
            [DetectionBox(bbox=(10, 40, 90, 120), label="bed", score=0.8)],
            metadata,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].bbox, (5, 5, 45, 45))

    def test_excludes_people_and_room_planes_by_default(self):
        image = Image.new("RGB", (20, 20), "white")
        person_mask = np.zeros((20, 20), dtype=bool)
        person_mask[2:10, 2:10] = True
        floor_mask = np.zeros((20, 20), dtype=bool)
        floor_mask[12:20, :] = True
        bed_mask = np.zeros((20, 20), dtype=bool)
        bed_mask[4:15, 6:18] = True

        result = build_result_from_annotations(
            image,
            [
                {"segmentation": person_mask, "label": "person", "score": 0.9},
                {"segmentation": floor_mask, "label": "floor", "score": 0.9},
                {"segmentation": bed_mask, "label": "bed", "score": 0.9},
            ],
            AutoSegmentationConfig(min_area_ratio=0.01),
            mode="hybrid",
        )

        self.assertEqual([instance.label for instance in result.instances], ["bed"])
        reasons = {entry["reason"] for entry in result.rejected}
        self.assertIn("excluded_person", reasons)
        self.assertIn("excluded_room_surface", reasons)
        self.assertTrue(result.quality_passed)

    def test_save_scene_input_can_write_debug_without_batch_masks(self):
        image = Image.new("RGB", (10, 10), "white")
        result = build_result_from_annotations(
            image,
            [],
            AutoSegmentationConfig(min_area_ratio=0.01),
            mode="hybrid",
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_scene_input(result, tmp, write_batch_files=False)
            names = {path.name for path in Path(tmp).iterdir()}
            debug_names = {path.name for path in (Path(tmp) / "_auto_segmentation").iterdir()}

        self.assertIn("scene.jpg", names)
        self.assertIn("masked_scene.png", names)
        self.assertNotIn("0.png", names)
        self.assertNotIn("0_mask.png", names)
        self.assertIn("manifest.json", debug_names)
        self.assertIn("segmentation_overlay.png", debug_names)
        self.assertIn("mask_contact_sheet.png", debug_names)

    def test_segment_image_routes_hybrid_boxes_without_loading_models(self):
        config = AutoSegmentationConfig(
            segmentation_mode="hybrid",
            image_preprocess="none",
            min_area_ratio=0.01,
        )

        class FakeDetector:
            def detect(self, image):
                return [DetectionBox(bbox=(2, 2, 12, 12), label="bed", score=0.8)]

        class FakeBoxSegmenter:
            def segment_boxes(self, image, boxes, mode, preprocessing=None, warnings=None):
                mask = np.zeros((20, 20), dtype=bool)
                left, top, right, bottom = boxes[0].bbox
                mask[top:bottom, left:right] = True
                return build_result_from_annotations(
                    image,
                    [{"segmentation": mask, "label": boxes[0].label, "score": boxes[0].score}],
                    config,
                    mode=mode,
                    preprocessing=preprocessing,
                    warnings=warnings,
                )

        result = segment_image(
            Image.new("RGB", (20, 20), "white"),
            config=config,
            detector=FakeDetector(),
            box_segmenter=FakeBoxSegmenter(),
        )

        self.assertEqual(result.mode, "hybrid")
        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.instances[0].label, "bed")

    def test_segment_image_falls_back_to_auto_masks_when_hybrid_has_no_boxes(self):
        config = AutoSegmentationConfig(
            segmentation_mode="hybrid",
            image_preprocess="none",
            min_area_ratio=0.01,
        )

        class EmptyDetector:
            def detect(self, image):
                return []

        class FakeAutoSegmenter:
            def segment(self, image):
                mask = np.zeros((20, 20), dtype=bool)
                mask[4:12, 4:12] = True
                return build_result_from_annotations(
                    image,
                    [{"segmentation": mask, "score": 0.8}],
                    config,
                    mode="sam2_auto",
                )

        result = segment_image(
            Image.new("RGB", (20, 20), "white"),
            config=config,
            detector=EmptyDetector(),
            auto_segmenter=FakeAutoSegmenter(),
        )

        self.assertEqual(result.mode, "sam2_auto_fallback")
        self.assertEqual(len(result.instances), 1)
        self.assertTrue(any("produced no boxes" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
