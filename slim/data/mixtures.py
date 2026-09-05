"""Named LIBERO dataset mixtures."""

LIBERO_ALL = [
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]

NAMED_MIXTURES = {
    "libero_all": LIBERO_ALL,
    "libero_all_90": [*LIBERO_ALL, "libero_90_no_noops_lerobot"],
}


def resolve_mixture(name: str) -> list[str]:
    try:
        return list(NAMED_MIXTURES[name])
    except KeyError as exc:
        choices = ", ".join(sorted(NAMED_MIXTURES))
        raise KeyError(f"Unknown data mixture {name!r}. Available: {choices}") from exc
