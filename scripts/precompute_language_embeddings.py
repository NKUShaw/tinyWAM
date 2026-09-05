#!/usr/bin/env python3
"""Precompute per-task T5 embeddings used by the canonical LIBERO recipe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from slim.data.mixtures import resolve_mixture


def _dataset_roots(data_root: Path, data_mix: str) -> list[Path]:
    try:
        names = resolve_mixture(data_mix)
    except KeyError:
        names = [name.strip() for name in data_mix.split(",") if name.strip()]
    return [data_root / name for name in names]


def _unique_tasks(dataset_root: Path) -> list[str]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks = set()
    with episodes_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            values = json.loads(line).get("tasks", [])
            if values:
                tasks.add(str(values[0]))
    return sorted(tasks)


def _encode(
    texts: list[str],
    tokenizer,
    encoder,
    *,
    batch_size: int,
    max_tokens: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    arrays = []
    index = {}
    offset = 0
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            for row, text in enumerate(batch):
                length = int(mask[row].sum().item())
                value = (
                    hidden[row, :length]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                arrays.append(value)
                index[text] = {"offset": offset, "length": length}
                offset += length
    width = int(encoder.config.d_model)
    embeddings = (
        np.concatenate(arrays, axis=0)
        if arrays
        else np.zeros((0, width), dtype=np.float16)
    )
    return embeddings, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-mix", default="libero_all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=os.environ.get("T5_MODEL_DIR"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.encoder_path is None:
        parser.error("--encoder-path is required when T5_MODEL_DIR is not set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.encoder_path), local_files_only=True
    )
    encoder = T5EncoderModel.from_pretrained(
        str(args.encoder_path), local_files_only=True
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_root in _dataset_roots(args.data_root, args.data_mix):
        texts = _unique_tasks(dataset_root)
        embeddings, index = _encode(
            texts,
            tokenizer,
            encoder,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            device=device,
        )
        prefix = args.output_dir / f"{dataset_root.name}_t5small_lang_emb"
        np.save(f"{prefix}.npy", embeddings)
        with Path(f"{prefix}.index.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "dataset_root": str(dataset_root),
                    "encoder_path": str(args.encoder_path),
                    "max_text_tokens": args.max_tokens,
                    "embedding_dim": int(embeddings.shape[1]),
                    "dtype": "float16",
                    "index": index,
                },
                file,
                ensure_ascii=False,
            )
        print(
            f"[done] {dataset_root.name}: tasks={len(texts)} "
            f"shape={tuple(embeddings.shape)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
