# Environments

SLIM uses separate environments for training, standard LIBERO evaluation, and
LIBERO-Plus evaluation. The two simulators both install a Python package named
`libero` and must not share an environment.

## Training

Use Python 3.12.9 and CUDA 12.4:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements.txt
pip install -e .
```

Clone the official DINOv2 repository under
`${TORCH_HOME}/hub/facebookresearch_dinov2_main`, then place
`dinov2_vitb14_pretrain.pth` in `${DINOV2_MODEL_DIR}`. The exact source-tree
and weight hashes used for the reported runs are recorded in
`reproducibility/source_manifest.json`.

## LIBERO

Use a separate Python 3.11.8 environment:

```bash
python3.11 -m venv .venv-libero
source .venv-libero/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-eval.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -e . --no-deps
cd /path/to/SLIM
pip install -e . --no-deps
```

`requirements-eval.txt` pins the reported environment, including Python-side
packages for PyTorch 2.7.1 with CUDA 12.6, MuJoCo 3.3.2, robosuite 1.4.0, and
NumPy 1.26.4. Installing LIBERO with `--no-deps` prevents its package metadata
from replacing these versions.

The 8-GPU launcher creates an isolated LIBERO configuration automatically. For
manual evaluation, create the same five-entry `config.yaml` before importing
LIBERO:

```bash
export SLIM_LIBERO_HOME=/path/to/LIBERO
export LIBERO_CONFIG_PATH=/path/to/libero-standard-config
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
from pathlib import Path

checkout = Path(os.environ["SLIM_LIBERO_HOME"]).resolve()
root = checkout / "libero" / "libero"
config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
values = {
    "assets": root / "assets",
    "bddl_files": root / "bddl_files",
    "benchmark_root": root,
    "datasets": checkout / "libero" / "datasets",
    "init_states": root / "init_files",
}
config.write_text(
    "".join(f'{key}: "{value}"\n' for key, value in values.items()),
    encoding="utf-8",
)
PY
```

PyTorch 2.6 and newer default `torch.load()` to `weights_only=True`, while
LIBERO initialization states contain NumPy objects. For a manual evaluation,
allow loading these trusted benchmark files:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The 8-GPU launcher sets this variable automatically.

## LIBERO-Plus

Create another Python 3.11.8 environment from `requirements-eval.txt`, install
LIBERO-Plus at revision
`4976dc30028e805ff8094b55501d532c48fec182`, and apply:

```bash
python3.11 -m venv .venv-libero-plus
source .venv-libero-plus/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-eval.txt
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus
git checkout 4976dc30028e805ff8094b55501d532c48fec182
pip install -e . --no-deps
cd /path/to/SLIM
pip install -e . --no-deps
```

For manual LIBERO-Plus evaluation, generate its config with the same snippet,
using `SLIM_LIBERO_HOME=/path/to/LIBERO-plus` and a separate
`LIBERO_CONFIG_PATH`.

Do not install the two environment repositories into the same virtual
environment; both expose the `libero` Python package.

For category-level reports, set:

```bash
export LIBERO_PLUS_CLASSIFICATION=/path/to/LIBERO-plus/libero/libero/benchmark/task_classification.json
```

The 8-GPU evaluation launcher accepts these environment paths without
installing either simulator into the other environment:

```bash
export SLIM_LIBERO_HOME=/path/to/LIBERO
export SLIM_PLUS_HOME=/path/to/LIBERO-plus
export SLIM_LIBERO_PYTHON=/path/to/libero-env/bin/python
export SLIM_PLUS_PYTHON=/path/to/libero-plus-env/bin/python
```

The launcher creates separate LIBERO configuration directories under the
evaluation output root, so the standard and Plus checkouts never share BDDL,
asset, or initialization-state paths. Set `EVAL_PHASE=standard` or
`EVAL_PHASE=plus` to rerun only one phase; the default is `all`.

Verify both evaluation environments before launching:

```bash
python - <<'PY'
import mujoco
import numpy
import robosuite
import torch

assert mujoco.__version__ == "3.3.2"
assert numpy.__version__ == "1.26.4"
assert robosuite.__version__ == "1.4.0"
assert torch.__version__.startswith("2.7.1")
print("evaluation environment is ready")
PY
```

## CALVIN

CALVIN evaluation runs in a separate Python 3.8 environment. Do not install
its legacy Torch 1.13 and Hydra 1.1 dependencies into the SLIM training
environment.

```bash
conda create -n slim-calvin python=3.8
conda activate slim-calvin
python -m pip install setuptools==57.5.0 wheel==0.44.0
python -m pip install -r requirements-calvin.txt

git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=/path/to/calvin
python -m pip install -e "$CALVIN_ROOT/calvin_env/tacto" --no-deps
python -m pip install -e "$CALVIN_ROOT/calvin_env" --no-deps
python -m pip install -e "$CALVIN_ROOT/calvin_models" --no-deps
```

The policy remains in the Python 3.12 SLIM environment. The Python 3.8
simulator imports only the CALVIN evaluation client from this checkout through
`PYTHONPATH` and queries the policy server over WebSocket.

Configure the simulator environment and SLIM checkout explicitly:

```bash
export SLIM_CALVIN_PYTHON=/path/to/slim-calvin/bin/python
export CALVIN_ROOT=/path/to/calvin
export CALVIN_LANG_DATASET=/path/to/calvin_ABC_D_lerobot
export PYTHONPATH=/path/to/SLIM
```

The public `lewislf/calvin-abc-lerobot-depth` dataset is a file-sharded
LeRobot v3 dataset. Set `CALVIN_VIDEO_BACKEND=torchvision_av` to consume its
videos directly with `scripts/train_calvin_pipeline.sh`. Its relative action
is stored in `action.rel`, not `action`; the pipeline detects this from
`meta/info.json`. Set `CALVIN_ACTION_KEY` only to override that detection.
Use `frames` only when the dataset root already contains extracted JPEGs.

Verify it with:

```bash
"$SLIM_CALVIN_PYTHON" -m pip check
"$SLIM_CALVIN_PYTHON" - <<'PY'
import calvin_agent
import calvin_env
import hydra
import msgpack
import numpy
import pybullet
import torch
import websockets

assert torch.__version__.startswith("1.13.1")
assert hydra.__version__ == "1.1.1"
assert numpy.__version__ == "1.24.4"
assert websockets.__version__ == "13.1"
print("CALVIN evaluation environment is ready")
PY
```
