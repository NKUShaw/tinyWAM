# Checkpoint Migration

The compatibility layer accepts historical run `config.yaml` files and maps
their framework, dataset, and trainer sections into the public SLIM schema.
Parameter-bearing module attributes are unchanged, so canonical checkpoints
load without key remapping.

Old experimental auxiliary heads are intentionally unsupported. A checkpoint
that contains parameters from one of those heads will fail strict loading
instead of silently dropping them.
