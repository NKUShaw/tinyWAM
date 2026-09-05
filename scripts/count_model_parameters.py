#!/usr/bin/env python3
"""Instantiate a SLIM config and report total/trainable parameters by module."""

from __future__ import annotations

import argparse
from pathlib import Path

from slim.config import load_config
from slim.model import SLIMModel


def count(module) -> tuple[int, int]:
    parameters = list(module.parameters())
    return sum(p.numel() for p in parameters), sum(
        p.numel() for p in parameters if p.requires_grad
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    model = SLIMModel(config=config)
    total, trainable = count(model)
    print(f"config={args.config}")
    print(f"total={total:,}")
    print(f"trainable={trainable:,}")
    print(f"frozen={total - trainable:,}")
    print("module\ttotal\ttrainable")
    for name, module in model.named_children():
        module_total, module_trainable = count(module)
        print(f"{name}\t{module_total:,}\t{module_trainable:,}")


if __name__ == "__main__":
    main()
