import os
os.environ['ATTN_BACKEND'] = 'xformers'     # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
from PIL import Image
from scenegen.pipelines import SceneGenImageToScenePipeline
import torch
import argparse
from easydict import EasyDict as edict
import sys
import numpy as np
import gradio
from gradio_litmodel3d import LitModel3D
from tqdm import tqdm
import contextlib
import io
import logging
import time
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_auto_labels(value: str | None):
    from scenegen.segmentation import DEFAULT_AUTO_LABELS

    if not value:
        return DEFAULT_AUTO_LABELS
    labels = tuple(label.strip() for label in value.split(",") if label.strip())
    return labels or DEFAULT_AUTO_LABELS


def iter_input_images(input_path: str):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported input image type: {path}")
        yield path
        return
    if path.is_dir():
        for candidate in sorted(path.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                yield candidate
        return
    raise FileNotFoundError(f"Input image path does not exist: {path}")


def prepare_auto_segmented_inputs(opt, test_image_dir: str):
    if not opt.input_image:
        raise ValueError("--input_image is required when --auto_segment is set")

    from scenegen.segmentation import AutoSegmentationConfig, load_boxes_json, save_scene_input, segment_image

    config = AutoSegmentationConfig(
        sam2_checkpoint=opt.sam2_checkpoint,
        sam2_model_cfg=opt.sam2_model_cfg,
        device=opt.sam2_device,
        segmentation_mode=opt.segmentation_mode,
        image_preprocess=opt.image_preprocess,
        segment_max_side=opt.segment_max_side,
        portrait_crop_ratio=opt.portrait_crop_ratio,
        points_per_side=opt.sam2_points_per_side,
        points_per_batch=opt.sam2_points_per_batch,
        pred_iou_thresh=opt.sam2_pred_iou_thresh,
        stability_score_thresh=opt.sam2_stability_score_thresh,
        min_mask_region_area=opt.sam2_min_mask_region_area,
        min_area_ratio=opt.auto_min_area_ratio,
        max_area_ratio=opt.auto_max_area_ratio,
        min_new_area_ratio=opt.auto_min_new_area_ratio,
        max_instances=opt.auto_max_instances,
        sort_by=opt.auto_sort_by,
        detector_id=opt.detector_id,
        detection_threshold=opt.detection_threshold,
        auto_labels=parse_auto_labels(opt.auto_labels),
        exclude_people=not opt.include_people,
        include_room_surfaces=opt.include_room_surfaces,
        allow_low_quality_masks=opt.allow_low_quality_masks,
        plane_area_ratio=opt.plane_area_ratio,
        edge_touch_ratio=opt.edge_touch_ratio,
    )
    boxes_by_scene = load_boxes_json(opt.boxes_json) if opt.boxes_json else {}
    os.makedirs(test_image_dir, exist_ok=True)

    prepared_count = 0
    allowed_scene_ids = set()
    for image_path in iter_input_images(opt.input_image):
        scene_id = image_path.stem
        scene_dir = Path(test_image_dir) / scene_id
        existing_masks = list(scene_dir.glob("*_mask.png"))
        if existing_masks and not opt.auto_segment_overwrite:
            print(f"Auto segmentation skipped for {scene_id}: masks already exist")
            allowed_scene_ids.add(scene_id)
            continue

        print(f"Auto segmenting {image_path} -> {scene_dir}")
        boxes = boxes_by_scene.get(scene_id) or boxes_by_scene.get(None)
        if opt.segmentation_mode == "manual_boxes" and boxes is None:
            print(f"Manual box segmentation skipped for {scene_id}: no boxes found in --boxes_json")
            continue

        result = segment_image(image_path, config=config, boxes=boxes)
        if len(result.instances) == 0:
            print(f"Auto segmentation found no usable instances for {image_path}; saving debug report only")
            save_scene_input(result, scene_dir, overwrite=opt.auto_segment_overwrite, write_batch_files=False)
            continue

        write_batch_files = result.quality_passed or opt.allow_low_quality_masks
        save_scene_input(
            result,
            scene_dir,
            overwrite=opt.auto_segment_overwrite,
            write_batch_files=write_batch_files,
        )
        if not write_batch_files:
            print(f"Segmentation quality failed for {scene_id}; saved debug report only")
            continue

        print(
            f"Prepared {len(result.instances)} instance masks for {scene_id} "
            f"using {result.mode}; quality_passed={result.quality_passed}"
        )
        prepared_count += 1
        allowed_scene_ids.add(scene_id)

    if prepared_count == 0:
        print("Auto segmentation did not prepare any new scenes")
    return allowed_scene_ids

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',type=str, default="assets", help='Directory to save the metadata')
    parser.add_argument('--set', type=str, default='test', help='Test set only')
    parser.add_argument('--model_name', type=str, default='SceneGen', help='Model name to use for evaluation')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for evaluation')
    parser.add_argument('--gradio', action='store_true', help='Run Gradio interface webpage for visualization')
    parser.add_argument('--auto_segment', action='store_true', help='Automatically create masks from --input_image before inference')
    parser.add_argument('--input_image', type=str, default=None, help='Single image or image directory to auto-segment')
    parser.add_argument('--auto_segment_overwrite', action='store_true', help='Overwrite existing auto-segmented scene inputs')
    parser.add_argument('--segmentation_mode', type=str, default='hybrid', choices=['hybrid', 'manual_boxes', 'sam2_auto'], help='Segmentation strategy used by --auto_segment')
    parser.add_argument('--boxes_json', type=str, default=None, help='Manual boxes JSON for --segmentation_mode manual_boxes')
    parser.add_argument('--auto_labels', type=str, default=None, help='Comma-separated zero-shot detector labels for hybrid segmentation')
    parser.add_argument('--detector_id', type=str, default='IDEA-Research/grounding-dino-tiny', help='Transformers zero-shot detector model for hybrid segmentation')
    parser.add_argument('--detection_threshold', type=float, default=0.25, help='Zero-shot detection threshold for hybrid segmentation')
    parser.add_argument('--include_people', action='store_true', help='Keep person detections instead of filtering them by default')
    parser.add_argument('--include_room_surfaces', action='store_true', help='Keep wall/floor/ceiling-style masks instead of filtering them by default')
    parser.add_argument('--allow_low_quality_masks', action='store_true', help='Write batch masks even when quality gates warn about low-quality masks')
    parser.add_argument('--segment_only', action='store_true', help='Only create segmentation inputs and debug reports; do not run SceneGen')
    parser.add_argument('--image_preprocess', type=str, default='auto', choices=['auto', 'none'], help='Normalize raw photos before segmentation')
    parser.add_argument('--segment_max_side', type=int, default=2048, help='Maximum image side used for segmentation preprocessing')
    parser.add_argument('--portrait_crop_ratio', type=float, default=1.2, help='Auto-crop portrait images at or above this height/width ratio')
    parser.add_argument('--auto_max_instances', type=int, default=16, help='Maximum automatic masks to keep per scene')
    parser.add_argument('--auto_min_area_ratio', type=float, default=0.002, help='Minimum mask area as a fraction of image area')
    parser.add_argument('--auto_max_area_ratio', type=float, default=0.85, help='Maximum mask area as a fraction of image area')
    parser.add_argument('--auto_min_new_area_ratio', type=float, default=0.35, help='Minimum unclaimed area ratio after resolving overlapping masks')
    parser.add_argument('--auto_sort_by', type=str, default='score', choices=['score', 'area'], help='Automatic mask ordering before overlap filtering')
    parser.add_argument('--plane_area_ratio', type=float, default=0.45, help='Reject unlabeled plane-like masks above this image area ratio')
    parser.add_argument('--edge_touch_ratio', type=float, default=0.55, help='Reject large masks touching this fraction of image edges')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoints/sam2-hiera-large/sam2_hiera_large.pt', help='SAM2 checkpoint for automatic segmentation')
    parser.add_argument('--sam2_model_cfg', type=str, default='configs/sam2/sam2_hiera_l.yaml', help='SAM2 model config for automatic segmentation')
    parser.add_argument('--sam2_device', type=str, default='cuda', help='Device used for SAM2 automatic segmentation')
    parser.add_argument('--sam2_points_per_side', type=int, default=32, help='SAM2 automatic mask points per side')
    parser.add_argument('--sam2_points_per_batch', type=int, default=64, help='SAM2 automatic mask points per batch')
    parser.add_argument('--sam2_pred_iou_thresh', type=float, default=0.8, help='SAM2 automatic mask predicted IoU threshold')
    parser.add_argument('--sam2_stability_score_thresh', type=float, default=0.95, help='SAM2 automatic mask stability threshold')
    parser.add_argument('--sam2_min_mask_region_area', type=int, default=100, help='SAM2 postprocessing threshold for small mask regions')
    opt = parser.parse_args(sys.argv[1:])
    opt = edict(vars(opt))

    test_image_dir = os.path.join(opt.output_dir, f'masked_images_{opt.set}')
    auto_segment_scene_ids = None
    if opt.auto_segment:
        auto_segment_scene_ids = prepare_auto_segmented_inputs(opt, test_image_dir)
        if opt.segment_only:
            sys.exit(0)
    assert os.path.exists(test_image_dir), f"Test image directory {test_image_dir} does not exist"

    scene_output_dir = os.path.join(opt.output_dir, f'scene_{opt.set}_{opt.model_name}')
    os.makedirs(scene_output_dir, exist_ok=True)

    scene_ids = os.listdir(test_image_dir)
    scene_ids = sorted(scene_ids)
    if auto_segment_scene_ids is not None:
        scene_ids = [sid for sid in scene_ids if sid in auto_segment_scene_ids]

    existing_ids = {f.split('.')[0] for f in os.listdir(scene_output_dir) if f.endswith('.glb')}
    scene_ids = [sid for sid in scene_ids if sid not in existing_ids]

    if opt.gpu_num > 1:
        scene_ids = scene_ids[opt.gpu_id::opt.gpu_num]

    if opt.model_name == 'SceneGen':
        pipeline = SceneGenImageToScenePipeline.from_pretrained("checkpoints/scenegen")
        pipeline.cuda()
    total_time = 0
    total_assets = 0
    for scene_id in tqdm(scene_ids, desc=f"Processing scenes on GPU {opt.gpu_id}", position=opt.gpu_id, leave=True):
        try:
            Scene_path = os.path.join(test_image_dir, scene_id)
            images_path = [
                image_name for image_name in os.listdir(Scene_path)
                if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" not in image_name
            ]
            images_path = sorted(images_path) 
            images = [
                Image.open(os.path.join(Scene_path, image_name))
                for image_name in images_path
            ] 
            mask_images_path = [
                image_name for image_name in os.listdir(Scene_path)
                if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" in image_name and "masked_scene" not in image_name
            ]

            mask_images_path = sorted(mask_images_path)
            mask_images = [
                Image.open(os.path.join(Scene_path, image_name))
                for image_name in mask_images_path
            ]

            if not os.path.exists(os.path.join(Scene_path, "scene.jpg")):
                print(f"Scene image not found in {Scene_path}, skipping...")
                continue

            scene_image = Image.open(os.path.join(Scene_path, "scene.jpg"))

            num_assets = len(mask_images_path)
            if num_assets == 0:
                print(f"No mask images found in {Scene_path}")
                continue
            if len(images) == 0:
                print(f"No images found in {Scene_path}")
                continue

            if opt.model_name == 'SceneGen':
                # Sort mask_images and images by mask size

                # Calculate the size (number of white pixels) of each mask
                mask_sizes = []
                for i, mask in enumerate(mask_images):
                    # Convert mask to numpy array and count non-zero (white) pixels
                    mask_array = np.array(mask)
                    size = np.count_nonzero(mask_array)
                    mask_sizes.append((i, size))

                query_asset_order = "largest"

                # Sort indices by mask size (descending order - largest first)
                sorted_indices = [idx for idx, _ in sorted(mask_sizes, key=lambda x: x[1], reverse=True)]
                if query_asset_order != "largest":
                    sorted_indices = sorted_indices[::-1]
                # Reorder the images and mask_images according to mask size
                mask_images = [mask_images[i] for i in sorted_indices]
                images = [images[i] for i in sorted_indices]

                # Compute the inverse permutation to restore the original order of images
                restore_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i])

                # Redirect stdout and stderr to prevent disrupting tqdm progress bar
                # Redirect stdout, stderr and suppress logging messages
                original_log_level = logging.root.level
                logging.root.setLevel(logging.ERROR)  # Suppress INFO and WARNING messages
                
                start_time = time.time()
                with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                    outputs = pipeline.run_scene(
                        image=images,
                        mask_image=mask_images,
                        scene_image=scene_image,
                        preprocess_image=True,
                        sparse_structure_sampler_params={
                                "steps": 25,
                                "cfg_strength": 5.0,
                                "cfg_interval": [0.5, 1.0],
                                "rescale_t": 3.0
                            },
                        slat_sampler_params={
                                "steps": 25,
                                "cfg_strength": 5.0,
                                "cfg_interval": [0.5, 1.0],
                                "rescale_t": 3.0
                            },
                        resorted_indices=restore_indices,
                    )
                    torch.cuda.empty_cache()
                end_time = time.time()
                
                # Restore original logging level
                logging.root.setLevel(original_log_level)

                scene = outputs["scene"]
                scene.export(os.path.join(scene_output_dir, f"{scene_id}.glb"))

                total_time += (end_time - start_time)
                total_assets += num_assets
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            continue
    
    if total_assets > 0:
        avg_time_per_asset = total_time / total_assets
        print(f"\nTotal assets processed: {total_assets}")
        print(f"Total generation time: {total_time:.2f} seconds")
        print(f"Average generation time per asset: {avg_time_per_asset:.2f} seconds")
    
    if opt.gradio:
        # Find common scene IDs between ground truth and generated scenes
        gt_scene_dir = os.path.join(opt.output_dir, f'scene_{opt.set}')
        gen_scene_dir = scene_output_dir

        if os.path.exists(gt_scene_dir):
            gt_scene_ids = [f.split('.')[0] for f in os.listdir(gt_scene_dir) if f.endswith('.glb')]
            gen_scene_ids = [f.split('.')[0] for f in os.listdir(gen_scene_dir) if f.endswith('.glb')]
            
            # Find scene IDs that appear in both directories
            common_scene_ids = sorted(list(set(gt_scene_ids) & set(gen_scene_ids)))
            print(f"Found {len(common_scene_ids)} common scenes between ground truth and generated results")
        else:
            print(f"Ground truth scene directory {gt_scene_dir} does not exist")
            common_scene_ids = []

        if common_scene_ids:
            with gradio.Blocks() as demo:
                gradio.Markdown("## 3D Scene Comparison Viewer")
                
                with gradio.Row():
                    scene_dropdown = gradio.Dropdown(
                        choices=common_scene_ids, 
                        label="Select Scene", 
                        value=common_scene_ids[0] if common_scene_ids else None
                    )
                    view_button = gradio.Button("View Scene")
                
                with gradio.Row():
                    with gradio.Column():
                        gradio.Markdown("### Ground Truth")
                        gt_model_output = LitModel3D(
                            label="Ground Truth Scene",
                            exposure=5.0,
                            height=500,
                            interactive=True,
                        )
                        gt_download_btn = gradio.DownloadButton(label="Download GT GLB", interactive=False)
                    
                    with gradio.Column():
                        gradio.Markdown(f"### Generated ({opt.model_name})")
                        gen_model_output = LitModel3D(
                            label="Generated Scene",
                            exposure=5.0,
                            height=500,
                            interactive=True,
                        )
                        gen_download_btn = gradio.DownloadButton(label="Download Generated GLB", interactive=False)
                
                def load_scenes(scene_id):
                    gt_path = os.path.join(gt_scene_dir, f"{scene_id}.glb")
                    gen_path = os.path.join(gen_scene_dir, f"{scene_id}.glb")
                    
                    gt_exists = os.path.exists(gt_path)
                    gen_exists = os.path.exists(gen_path)
                    
                    return (
                        gt_path if gt_exists else None,
                        gradio.update(interactive=gt_exists, value=gt_path if gt_exists else None),
                        gen_path if gen_exists else None,
                        gradio.update(interactive=gen_exists, value=gen_path if gen_exists else None),
                    )
                    
                view_button.click(
                    load_scenes,
                    inputs=[scene_dropdown],
                    outputs=[gt_model_output, gt_download_btn, gen_model_output, gen_download_btn],
                )
                    
            demo.launch(share=True)
            print("Gradio interface launched for scene comparison")
        else:
            print("No common scenes found for visualization")
        
