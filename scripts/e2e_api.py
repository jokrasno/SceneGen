import contextlib
import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from scenegen.pipelines import SceneGenImageToScenePipeline
from scenegen.segmentation import AutoSegmentationConfig, DEFAULT_AUTO_LABELS, save_scene_input, segment_image
from scenegen.utils.inference_scene import run_scene, seg_image_to_label_map


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "test_runs" / "e2e_api"
INPUT_ROOT = RUN_ROOT / "inputs"
OUTPUT_ROOT = RUN_ROOT / "outputs"
CHECKPOINT = ROOT / "checkpoints" / "scenegen"
PUBLIC_BASE_URL = os.environ.get("SCENEGEN_PUBLIC_BASE_URL", "https://scenegen-gpu.tail948ef9.ts.net")

GYM_AND_ROOM_LABELS = tuple(
    dict.fromkeys(
        DEFAULT_AUTO_LABELS
        + (
            "treadmill",
            "exercise bike",
            "elliptical machine",
            "weight bench",
            "bench",
            "dumbbell",
            "barbell",
            "kettlebell",
            "weight rack",
            "cable machine",
            "rowing machine",
            "gym mat",
            "medicine ball",
            "locker",
            "fan",
            "speaker",
            "trash can",
        )
    )
)


app = FastAPI(title="SceneGen E2E API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
pipeline_lock = threading.Lock()
pipeline: SceneGenImageToScenePipeline | None = None


def _job_url(job_id: str, path: str) -> str:
    return f"{PUBLIC_BASE_URL.rstrip('/')}/v1/jobs/{job_id}/{path.lstrip('/')}"


def _set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        jobs[job_id].update(updates)
        jobs[job_id]["updated_at"] = time.time()


def _get_pipeline() -> SceneGenImageToScenePipeline:
    global pipeline
    with pipeline_lock:
        if pipeline is None:
            loaded = SceneGenImageToScenePipeline.from_pretrained(str(CHECKPOINT))
            loaded.cuda()
            pipeline = loaded
        return pipeline


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)


def _load_label_map(path: Path) -> Image.Image:
    label_map = seg_image_to_label_map(Image.open(path))
    if len(np.unique(np.asarray(label_map))) <= 1:
        raise ValueError("label_map contains no non-background object labels")
    return label_map


def _run_job(
    job_id: str,
    image_path: Path,
    label_map_path: Path | None,
    segmentation_mode: Literal["hybrid", "sam2_auto"],
    max_instances: int,
    allow_low_quality_masks: bool,
    include_people: bool,
    seed: int,
    positions_type: Literal["avg", "last"],
    simplify: float,
    texture_size: int,
) -> None:
    job_dir = OUTPUT_ROOT / job_id
    scene_input_dir = job_dir / "scene_input"
    glb_path = job_dir / "scene.glb"
    manifest_path = job_dir / "manifest.json"
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        _set_job(job_id, status="running", stage="loading_image")
        rgb_image = Image.open(image_path).convert("RGB")

        if label_map_path is not None:
            _set_job(job_id, stage="using_uploaded_label_map")
            label_map = _load_label_map(label_map_path)
            if label_map.size != rgb_image.size:
                label_map = label_map.resize(rgb_image.size, Image.Resampling.NEAREST)
            segmentation_summary = {
                "mode": "uploaded_label_map",
                "instances": int(len(np.unique(np.asarray(label_map))) - 1),
                "quality_passed": True,
                "warnings": [],
            }
        else:
            _set_job(job_id, stage="segmenting")
            config = AutoSegmentationConfig(
                segmentation_mode=segmentation_mode,
                auto_labels=GYM_AND_ROOM_LABELS,
                max_instances=max_instances,
                allow_low_quality_masks=allow_low_quality_masks,
                exclude_people=not include_people,
                points_per_side=48,
                points_per_batch=64,
                pred_iou_thresh=0.75,
                stability_score_thresh=0.9,
                min_area_ratio=0.0015,
                min_new_area_ratio=0.25,
            )
            result = segment_image(image_path, config=config)
            if len(result.instances) == 0:
                save_scene_input(result, scene_input_dir, overwrite=True, write_batch_files=False)
                raise ValueError("automatic segmentation found no usable object instances")
            save_scene_input(
                result,
                scene_input_dir,
                overwrite=True,
                write_batch_files=result.quality_passed or allow_low_quality_masks,
            )
            rgb_image = result.image
            label_map = result.label_map
            segmentation_summary = {
                "mode": result.mode,
                "instances": len(result.instances),
                "quality_passed": result.quality_passed,
                "warnings": result.warnings,
                "debug": {
                    "overlay_url": _job_url(job_id, "segmentation_overlay.png"),
                    "contact_sheet_url": _job_url(job_id, "mask_contact_sheet.png"),
                    "manifest_url": _job_url(job_id, "auto_segmentation_manifest.json"),
                },
            }

        _set_job(job_id, stage="generating_scene", segmentation=segmentation_summary)
        pipe = _get_pipeline()
        original_log_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            scene = run_scene(
                pipe,
                rgb_image,
                label_map,
                seed=seed,
                ss_num_inference_steps=25,
                ss_cfg_strength=5.0,
                ss_cfg_interval=[0.5, 1.0],
                ss_rescale_t=3.0,
                slat_num_inference_steps=25,
                slat_cfg_strength=5.0,
                slat_cfg_interval=[0.5, 1.0],
                slat_rescale_t=3.0,
                order="largest",
                positions_type=positions_type,
                simplify=simplify,
                texture_size=texture_size,
            )
        logging.root.setLevel(original_log_level)
        torch.cuda.empty_cache()

        _set_job(job_id, stage="exporting")
        scene.export(glb_path)
        manifest = {
            "job_id": job_id,
            "image": str(image_path),
            "glb": str(glb_path),
            "segmentation": segmentation_summary,
            "settings": {
                "seed": seed,
                "positions_type": positions_type,
                "simplify": simplify,
                "texture_size": texture_size,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _set_job(
            job_id,
            status="complete",
            stage="complete",
            glb_path=str(glb_path),
            glb_url=_job_url(job_id, "scene.glb"),
            manifest_url=_job_url(job_id, "manifest.json"),
        )
    except Exception as exc:
        torch.cuda.empty_cache()
        _set_job(job_id, status="failed", stage="failed", error=str(exc))


@app.get("/")
def root():
    return {
        "service": "SceneGen E2E API",
        "docs": f"{PUBLIC_BASE_URL.rstrip('/')}/docs",
        "create_job": "POST /v1/jobs multipart/form-data field image=<file>",
    }


@app.get("/health")
def health():
    return {"ok": True, "queue_workers": 1, "output_root": str(OUTPUT_ROOT)}


@app.post("/v1/jobs")
def create_job(
    image: UploadFile = File(...),
    label_map: UploadFile | None = File(None),
    segmentation_mode: Literal["hybrid", "sam2_auto"] = Form("hybrid"),
    max_instances: int = Form(12),
    allow_low_quality_masks: bool = Form(True),
    include_people: bool = Form(False),
    seed: int = Form(0),
    positions_type: Literal["avg", "last"] = Form("avg"),
    simplify: float = Form(0.95),
    texture_size: int = Form(1024),
):
    if max_instances < 1 or max_instances > 64:
        raise HTTPException(status_code=400, detail="max_instances must be between 1 and 64")
    if texture_size not in {512, 1024, 2048, 4096}:
        raise HTTPException(status_code=400, detail="texture_size must be one of 512, 1024, 2048, 4096")

    job_id = uuid.uuid4().hex[:12]
    input_dir = INPUT_ROOT / job_id
    suffix = Path(image.filename or "image.jpg").suffix.lower() or ".jpg"
    image_path = input_dir / f"input{suffix}"
    _save_upload(image, image_path)

    label_map_path = None
    if label_map is not None and label_map.filename:
        label_suffix = Path(label_map.filename).suffix.lower() or ".png"
        label_map_path = input_dir / f"label_map{label_suffix}"
        _save_upload(label_map, label_map_path)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "status_url": _job_url(job_id, ""),
        }

    executor.submit(
        _run_job,
        job_id,
        image_path,
        label_map_path,
        segmentation_mode,
        max_instances,
        allow_low_quality_masks,
        include_people,
        seed,
        positions_type,
        simplify,
        texture_size,
    )
    return jobs[job_id]


@app.get("/v1/jobs")
def list_jobs():
    with jobs_lock:
        return {"jobs": sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job


@app.get("/v1/jobs/{job_id}/scene.glb")
def get_scene(job_id: str):
    path = OUTPUT_ROOT / job_id / "scene.glb"
    if not path.exists():
        raise HTTPException(status_code=404, detail="scene not ready")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


@app.head("/v1/jobs/{job_id}/scene.glb")
def head_scene(job_id: str):
    path = OUTPUT_ROOT / job_id / "scene.glb"
    if not path.exists():
        raise HTTPException(status_code=404, detail="scene not ready")
    return Response(
        status_code=200,
        media_type="model/gltf-binary",
        headers={
            "content-length": str(path.stat().st_size),
            "content-disposition": f'attachment; filename="{job_id}.glb"',
        },
    )


@app.get("/v1/jobs/{job_id}/manifest.json")
def get_manifest(job_id: str):
    path = OUTPUT_ROOT / job_id / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="manifest not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/v1/jobs/{job_id}/auto_segmentation_manifest.json")
def get_auto_segmentation_manifest(job_id: str):
    path = OUTPUT_ROOT / job_id / "scene_input" / "_auto_segmentation" / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="auto segmentation manifest not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/v1/jobs/{job_id}/segmentation_overlay.png")
def get_segmentation_overlay(job_id: str):
    path = OUTPUT_ROOT / job_id / "scene_input" / "_auto_segmentation" / "segmentation_overlay.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="segmentation overlay not ready")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/jobs/{job_id}/mask_contact_sheet.png")
def get_mask_contact_sheet(job_id: str):
    path = OUTPUT_ROOT / job_id / "scene_input" / "_auto_segmentation" / "mask_contact_sheet.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="mask contact sheet not ready")
    return FileResponse(path, media_type="image/png")
