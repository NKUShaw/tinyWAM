"""Optimizer grouping and CLI override helpers."""

from __future__ import annotations


def normalize_overrides(arguments: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item.startswith("--"):
            key = item[2:]
            if "=" in key:
                normalized.append(key)
            elif index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                normalized.append(f"{key}={arguments[index + 1]}")
                index += 1
            else:
                normalized.append(f"{key}=true")
        else:
            normalized.append(item)
        index += 1
    return normalized


def _uses_no_decay(name: str) -> bool:
    return name.endswith(".bias") or any(
        pattern in name
        for pattern in ("norm.weight", "norm.bias", "ln_", ".ln.", "layernorm")
    )


def _split_decay_groups(named_parameters, lr: float, name: str, weight_decay: float):
    decay = []
    no_decay = []
    for parameter_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        target = no_decay if _uses_no_decay(parameter_name) else decay
        target.append(parameter)

    groups = []
    if decay:
        groups.append(
            {
                "name": name,
                "params": decay,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            }
        )
    if no_decay:
        groups.append(
            {
                "name": f"{name}_no_decay",
                "params": no_decay,
                "lr": float(lr),
                "weight_decay": 0.0,
            }
        )
    return groups


def build_parameter_groups(model, config):
    rates = config.training.learning_rate
    weight_decay = float(config.training.optimizer.weight_decay)
    named_groups = {"base": [], "action_model": [], "vision_encoder": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vision_encoder."):
            group_name = "vision_encoder"
        elif name.startswith("action_model."):
            group_name = "action_model"
        else:
            group_name = "base"
        named_groups[group_name].append((name, parameter))

    groups = []
    for group_name in ("action_model", "vision_encoder", "base"):
        groups.extend(
            _split_decay_groups(
                named_groups[group_name],
                lr=float(rates[group_name]),
                name=group_name,
                weight_decay=weight_decay,
            )
        )
    return groups
