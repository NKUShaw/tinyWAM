# Data and model assets

## LIBERO

The reported recipe uses no-op-filtered **LeRobot v2.1** conversions from the
[IPEC-COMMUNITY LIBERO collection](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset).
It does not use the original LIBERO HDF5 files or a LeRobot v3 conversion.

| Suite | Hugging Face repository | Episodes | Frames |
| --- | --- | ---: | ---: |
| Object | `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` | 454 | 66,984 |
| Goal | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` | 428 | 52,042 |
| Spatial | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` | 432 | 52,970 |
| LIBERO-10 | `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` | 379 | 101,469 |
| LIBERO-90 | `IPEC-COMMUNITY/libero_90_no_noops_lerobot` | 3,921 | 569,249 |

Download the five repositories into the directory names expected by the
configs:

```bash
python -m pip install huggingface_hub
mkdir -p "$LIBERO_DATA_ROOT"
for repo in \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_90_no_noops_lerobot
do
  hf download "$repo" \
    --repo-type dataset \
    --local-dir "$LIBERO_DATA_ROOT/${repo#*/}"
done
```

Each dataset reports `codebase_version: v2.1` and `fps: 20`. The two camera
keys are `observation.images.image` and `observation.images.wrist_image`, both
stored as 256x256 AV1 video. Actions are 7D and `observation.state` is 8D; the
canonical config consumes the first seven state values. Images are resized to
224x224 by the model.

Stage 1 uses `libero_all_90` (all five datasets). Stage 2 uses `libero_all`
(Object, Goal, Spatial, and LIBERO-10). For parity with the reported data
loading, Goal episode 82 is excluded.

The default configs decode the original videos with `torchvision_av`. For a
frame-backed ablation, extract JPEGs once:

```bash
python scripts/extract_lerobot_frames.py --workers 16 \
  "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot" \
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot" \
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot" \
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot" \
  "$LIBERO_DATA_ROOT/libero_90_no_noops_lerobot"
```

Language can be encoded online. To precompute the float16 T5 cache for
`libero_all`:

```bash
python scripts/precompute_language_embeddings.py \
  --data-root "$LIBERO_DATA_ROOT" \
  --data-mix libero_all \
  --output-dir "$SLIM_LANGUAGE_CACHE"
```

## CALVIN

The source demonstrations are distributed by
[`mees/calvin`](https://github.com/mees/calvin). A compatible community-hosted
LeRobot v3 conversion is available at
[`lewislf/calvin-abc-lerobot-depth`](https://huggingface.co/datasets/lewislf/calvin-abc-lerobot-depth):

```bash
export CALVIN_LANG_DATASET=/path/to/calvin_ABC_D_lerobot
hf download lewislf/calvin-abc-lerobot-depth \
  --repo-type dataset \
  --local-dir "$CALVIN_LANG_DATASET"
```

The recipe uses 7D relative actions, 15D robot state, two cameras
(`observation.images.rgb_static` and `observation.images.rgb_gripper`), and an
action horizon of 12. In the public v3 conversion, the relative action is
stored in `action.rel`; the pipeline detects this from `meta/info.json`. Set
`CALVIN_ACTION_KEY` only when overriding that detection.

The public conversion is video-sharded. Set
`CALVIN_VIDEO_BACKEND=torchvision_av` to decode it directly. The default
frame backend is retained for parity with the original per-episode conversion.

## Model assets

SLIM uses DINOv2-B/14 and T5-small. Keep both assets local and point
`DINOV2_MODEL_DIR` and `T5_MODEL_DIR` to them. The source-tree and weight hashes
used during migration are recorded in
[`reproducibility/source_manifest.json`](../reproducibility/source_manifest.json).
