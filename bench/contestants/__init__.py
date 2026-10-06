"""Contestants by name. Building one loads nothing; each model loads on its first prediction."""

from __future__ import annotations

from bench.contestants.base import Contestant, effective_scores, map_labels  # noqa: F401
from bench.contestants.detoxify import DetoxifyContestant
from bench.contestants.hf import horizon_mmbert
from bench.contestants.laya import laya_app, laya_taxonomy

FACTORIES = {
    "laya": lambda device: laya_taxonomy(device=device),
    "laya-app": lambda device: laya_app(device=device),
    "detoxify": lambda device: DetoxifyContestant(device=device),
    "horizon-mmbert": lambda device: horizon_mmbert(device=device),
}
DEFAULT_CONTESTANTS = tuple(FACTORIES)
# Only open, self-hostable models: no paid or hosted APIs, and no message leaves our machines.
OPEN_LICENSES = ("Apache-2.0", "MIT")


def build_contestants(names, device: str | None = None, factories: dict | None = None) -> list:
    factories = FACTORIES if factories is None else factories
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"modelo desconocido: {', '.join(unknown)}. Opciones: {', '.join(factories)}")
    return [factories[n](device) for n in names]
