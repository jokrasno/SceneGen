import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scenegen.segmentation import AutoSegmentationConfig, build_result_from_annotations, save_scene_input


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


if __name__ == "__main__":
    unittest.main()
