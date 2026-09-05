# Ablations

The files in this directory modify one factor relative to the canonical
Stage 1 and Stage 2 recipes:

- `stage1_idm025_fdm1_h8.yaml`: raises the IDM weight from 0.125 to 0.25.
- `stage1_idm_only_h8.yaml`: inverse dynamics only.
- `stage1_fdm_only_h8.yaml`: future dynamics only.
- `stage1_idm1_fdm1_h8.yaml`: equal objective weights.
- `stage1_no_ema_h8.yaml`: disables the target encoder EMA.
- `stage2_with_ema_h8_40ep.yaml`: enables EMA during policy training.
- `stage2_without_stage1_h8_43ep.yaml`: compute-matched direct policy training.

Use OmegaConf overrides for values not repeated in these small overlays.
