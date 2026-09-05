# SLIM

**SLIM: Self-supervised Latent Interaction Model** is a compact,
language-conditioned policy for robot manipulation. It learns action-grounded
predictive representations before optimizing a flow-matching control policy.

SLIM combines a DINOv2 vision encoder, T5 language embeddings, and the
**SLIM Transformer**, a two-stream transformer for observation and action
tokens.

## Method

Training has two stages:

1. **Action-Grounded Masked Trajectory Prediction** learns inverse dynamics
   (IDM) and future dynamics (FDM) from current/future observations and action
   chunks.
2. **Flow-Matching Policy Training** learns the control policy from current
   observations, language, proprioception, and demonstrations.

The default LIBERO recipe uses:

- Stage 1: LIBERO all+90, IDM:FDM = `0.125:1`, H8, 3 epochs, EMA enabled.
- Stage 2: LIBERO all, policy-only, H8, 40 epochs, EMA disabled.
- Both stages: original LeRobot videos decoded with `torchvision_av`, global
  batch size 128, and bias/norm parameters excluded from weight decay.

## Installation

The reported training environment used Python 3.12.9 and CUDA 12.4.

```bash
git clone https://github.com/kzz1031/SLIM.git
cd SLIM
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download DINOv2-B/14 and T5-small locally, then configure machine-specific
paths:

```bash
export LIBERO_DATA_ROOT=/path/to/lerobot/libero
export DINOV2_MODEL_DIR=/path/to/dinov2-vitb14
export T5_MODEL_DIR=/path/to/t5-small
export SLIM_CACHE_DIR=/path/to/cache
export SLIM_LANGUAGE_CACHE=/path/to/language_embeddings  # optional
```

`DINOV2_MODEL_DIR` must contain `dinov2_vitb14_pretrain.pth`, and the DINOv2
source checkout must be available at
`${TORCH_HOME:-$HOME/.cache/torch}/hub/facebookresearch_dinov2_main`.
`T5_MODEL_DIR` must contain a local T5-small checkpoint.

See [docs/data.md](docs/data.md) for dataset downloads and schemas, and
[docs/environments.md](docs/environments.md) for the exact training and
evaluation environments.

## Training

The launch helpers use eight GPUs by default. Override `NPROC_PER_NODE` and
`CUDA_VISIBLE_DEVICES` when using a different topology; adjust per-device batch
size if the global batch size should remain 128.

### Stage 1

```bash
export WANDB_MODE=offline  # optional
bash scripts/train_stage1_8gpu.sh
```

Equivalent command:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m slim.training.stage1 \
  --config configs/libero/stage1_idm0125_fdm1_h8.yaml
```

### Stage 2

```bash
bash scripts/train_stage2_8gpu.sh \
  checkpoints/stage1/<run>/checkpoints/epoch_3_pytorch_model.pt
```

Equivalent command:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m slim.training.stage2 \
  --config configs/libero/stage2_policy_h8_40ep.yaml \
  --init-checkpoint checkpoints/stage1/<run>/checkpoints/epoch_3_pytorch_model.pt
```

`--init-checkpoint` loads model weights from Stage 1. To continue an
interrupted distributed run with optimizer, scheduler, RNG, and data position
intact, use the same process topology and pass the saved state:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m slim.training.stage2 \
  --config configs/libero/stage2_policy_h8_40ep.yaml \
  --resume-state checkpoints/stage2/<run>/states/step_00005000
```

Use `--auto-resume` with `--run.timestamp=false --run.name=<existing-run>` to
select the latest state in an existing run. Any config value can be overridden
with a dotted CLI argument, for example `--training.max_epochs=5`.

Alternative objectives, EMA settings, and the direct policy baseline are under
[`configs/libero/ablations`](configs/libero/ablations/README.md). Explicit
frame-backed configs are provided for controlled input-pipeline ablations.

## Evaluation

Standard LIBERO and LIBERO-Plus must use separate Python 3.11 environments
because both install a package named `libero`. Follow
[docs/environments.md](docs/environments.md), then configure:

```bash
export SLIM_TRAIN_PYTHON=/path/to/slim-training-env/bin/python
export SLIM_LIBERO_PYTHON=/path/to/libero-mujoco332-env/bin/python
export SLIM_PLUS_PYTHON=/path/to/libero-plus-mujoco332-env/bin/python
export SLIM_LIBERO_HOME=/path/to/LIBERO
export SLIM_PLUS_HOME=/path/to/LIBERO-plus
export LIBERO_PLUS_CLASSIFICATION="$SLIM_PLUS_HOME/libero/libero/benchmark/task_classification.json"
```

Run the complete 8-GPU protocol:

```bash
bash scripts/evaluate_all_8gpu.sh \
  /path/to/epoch_40_pytorch_model.pt \
  outputs/canonical_eval \
  12000
```

The optional final argument is the first of eight consecutive policy-server
ports. The launcher waits for a stable checkpoint and eight idle GPUs, then
runs standard LIBERO followed by LIBERO-Plus. It disables video recording by
default, supports marker-level LIBERO-Plus resume, and writes:

```text
outputs/canonical_eval/
  libero/summary.txt
  libero_plus/summary.txt
```

Set `EVAL_PHASE=standard` or `EVAL_PHASE=plus` to run one phase. Both reported
protocols send the legacy 7D proprioceptive state; `--no-send-state` is reserved
for an explicit state ablation.

For a single standard LIBERO suite, start a policy server:

```bash
python -m slim.serving.server \
  --checkpoint /path/to/epoch_40_pytorch_model.pt \
  --port 10093 \
  --bf16
```

Then run the client in the standard LIBERO environment:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m slim.evaluation.libero.evaluate \
  --checkpoint /path/to/epoch_40_pytorch_model.pt \
  --host 127.0.0.1 \
  --port 10093 \
  --task_suite_name libero_10 \
  --num_trials_per_task 50 \
  --action_chunk_size 8 \
  --send-state
```

## CALVIN

CALVIN uses the same training environment but a separate Python 3.8 simulator
environment. Set the model, dataset, and cache paths described in
[docs/data.md](docs/data.md) and [docs/environments.md](docs/environments.md),
then run:

```bash
bash scripts/train_calvin_pipeline.sh
```

For the public LeRobot v3 conversion, select direct video decoding:

```bash
export CALVIN_VIDEO_BACKEND=torchvision_av
bash scripts/train_calvin_pipeline.sh
```

Evaluate a Stage 2 checkpoint with the Python 3.8 CALVIN client:

```bash
export PYTHONPATH=/path/to/SLIM
export PYOPENGL_PLATFORM=egl

"$SLIM_CALVIN_PYTHON" -m slim.evaluation.calvin.evaluate \
  --dataset-path "$CALVIN_ROOT/dataset/task_ABC_D" \
  --action-stats /path/to/stage2/run/action_stats_calvin_ABC_D_lerobot.json \
  --dataset-name calvin_ABC_D_lerobot \
  --action-horizon 12 \
  --exec-stride 12 \
  --host 127.0.0.1 \
  --port 10093 \
  --output-dir evaluation_outputs/calvin
```

The multi-checkpoint watcher requires an explicit Stage 2 run directory:

```bash
bash scripts/watch_calvin_eval_5ep.sh checkpoints/stage2/<run>
```

## Reproducibility

`scripts/validate_parity.py` compares this package with the historical source
checkout for dataset samples, model keys, latent tensors, losses, and action
predictions. FP32 and BF16 tolerances are `1e-6` and `1e-3`.

`reproducibility/source_manifest.json` records the migration source revision,
runtime hashes, environment versions, and historical reference checkpoint. It
is a migration record, not the definition of the current default recipe.
Historical checkpoints are loaded through `slim.compat.legacy`; unsupported
experimental auxiliary heads fail strict loading.

## License

SLIM includes code derived from StarVLA and NVIDIA components. The upstream
terms and attribution notices are preserved in [LICENSE](LICENSE),
[NOTICE](NOTICE), and the relevant source files. Review those terms before
redistribution.
