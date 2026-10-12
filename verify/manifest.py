"""Arrangement manifest -> episode specs + scoring-only ground truth (see verify/README.md section 4).

The split is by arrangement only: every episode of an arrangement (all conditions, all lightings)
lands in the same split. Ground truth is returned separately and never goes to ReplayEnv or a policy.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from verify.schema import (
    CONDITIONS,
    Condition,
    EpisodeSpec,
    GroundTruth,
    Lighting,
    ManifestError,
    Split,
    split_arrangements,
    truth_from_condition,
)

MANIFEST_SCHEMA = "cmpe249.arrangements.v1"
RE_ARRANGEMENT = re.compile(r"^[A-Z][A-Z0-9]{1,15}$")  # same as capture/pi_capture.py
_COND_ORDER = {c: i for i, c in enumerate(CONDITIONS)}
_LIGHT_ORDER = {l: i for i, l in enumerate(Lighting)}


@dataclass(frozen=True)
class ArrangementRecord:
    arrangement_id: str
    target_id: str
    split: Split
    lightings: tuple[Lighting, ...] = (Lighting.L1, Lighting.L2)
    indistinguishable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", Split(self.split))
        lights = tuple(Lighting(l) for l in self.lightings)
        if not lights or len(set(lights)) != len(lights):
            raise ManifestError(f"{self.arrangement_id}: lightings must be non-empty and unique")
        object.__setattr__(self, "lightings", tuple(sorted(lights, key=_LIGHT_ORDER.__getitem__)))
        if not RE_ARRANGEMENT.match(self.arrangement_id):
            raise ManifestError(f"bad arrangement id {self.arrangement_id!r} (expected e.g. A01)")

    def to_dict(self) -> dict:
        return {"arrangement_id": self.arrangement_id, "target_id": self.target_id,
                "split": self.split.value, "lightings": [l.value for l in self.lightings],
                "indistinguishable": self.indistinguishable}

    @classmethod
    def from_dict(cls, d: dict) -> ArrangementRecord:
        return cls(d["arrangement_id"], d["target_id"], Split(d["split"]),
                   tuple(Lighting(l) for l in d.get("lightings", ("L1", "L2"))),
                   bool(d.get("indistinguishable", False)))


def _sort_key(s: EpisodeSpec) -> tuple:
    return (s.arrangement_id, _COND_ORDER[s.condition], _LIGHT_ORDER[s.lighting])


def episodes_from_arrangements(arrs: Sequence[ArrangementRecord]) -> tuple[list[EpisodeSpec], GroundTruth]:
    """3 conditions x lightings per arrangement, sorted; truth from condition (scoring only)."""
    specs = [EpisodeSpec(a.arrangement_id, c, l, a.target_id, a.split)
             for a in arrs for c in CONDITIONS for l in a.lightings]
    specs.sort(key=_sort_key)
    truth = GroundTruth.from_specs(specs, [a.arrangement_id for a in arrs if a.indistinguishable])
    validate(specs, truth)
    return specs, truth


def save_manifest(arrs: Sequence[ArrangementRecord], path: str | Path) -> None:
    episodes_from_arrangements(arrs)  # validates before writing
    doc = {"schema": MANIFEST_SCHEMA, "arrangements": [a.to_dict() for a in arrs]}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def load_arrangements(path: str | Path) -> list[ArrangementRecord]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: schema is {doc.get('schema')!r}, expected {MANIFEST_SCHEMA!r}")
    return [ArrangementRecord.from_dict(d) for d in doc["arrangements"]]


def load_manifest(path: str | Path) -> tuple[list[EpisodeSpec], GroundTruth]:
    return episodes_from_arrangements(load_arrangements(path))


def assign_splits(arrangement_ids: Sequence[str], n_dev: int, seed: int) -> dict[str, Split]:
    return split_arrangements(arrangement_ids, n_dev, seed)


def validate(specs: Sequence[EpisodeSpec], truth: GroundTruth | None = None) -> None:
    """Raise ManifestError on any protocol violation (duplicates, mixed splits, incomplete arrangements, truth)."""
    ids = [s.episode_id for s in specs]
    dup = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dup:
        raise ManifestError(f"duplicate episode ids: {dup}")
    by_arr: dict[str, list[EpisodeSpec]] = {}
    for s in specs:
        if not RE_ARRANGEMENT.match(s.arrangement_id):
            raise ManifestError(f"bad arrangement id {s.arrangement_id!r} (expected e.g. A01)")
        by_arr.setdefault(s.arrangement_id, []).append(s)
    for arr, group in by_arr.items():
        splits = {s.split for s in group}
        if len(splits) > 1:
            raise ManifestError(f"arrangement {arr} appears in more than one split: "
                                f"{sorted(x.value for x in splits)} (split is by arrangement only)")
        targets = {s.target_id for s in group}
        if len(targets) > 1:
            raise ManifestError(f"arrangement {arr} has more than one target_id: {sorted(targets)}")
        lights = {s.lighting for s in group}
        have = {(s.condition, s.lighting) for s in group}
        want = {(c, l) for c in CONDITIONS for l in lights}
        if have != want:
            missing = sorted(f"{c.value}-{l.value}" for c, l in want - have)
            raise ManifestError(f"arrangement {arr} is incomplete: missing {missing} "
                                f"(need visible/occluded/removed at every lighting)")
    if truth is None:
        return
    if set(truth.states) != set(ids):
        extra, missing = sorted(set(truth.states) - set(ids)), sorted(set(ids) - set(truth.states))
        raise ManifestError(f"truth keys differ from spec ids: extra {extra}, missing {missing}")
    for s in specs:
        if truth[s.episode_id] is not truth_from_condition(s.condition):
            raise ManifestError(f"truth for {s.episode_id} is {truth[s.episode_id].value}, "
                                f"condition says {truth_from_condition(s.condition).value}")
        if s.condition is Condition.VISIBLE and truth.is_indistinguishable(s.episode_id):
            raise ManifestError(f"{s.episode_id} is visible but marked indistinguishable")


def filter_split(specs: Sequence[EpisodeSpec], split: Split) -> list[EpisodeSpec]:
    split = Split(split)
    return [s for s in specs if s.split is split]


def summary(specs: Sequence[EpisodeSpec]) -> dict[str, int]:
    arr_split = {s.arrangement_id: s.split for s in specs}
    return {"n_arrangements": len(arr_split), "n_episodes": len(specs),
            "n_dev_arrangements": sum(v is Split.DEV for v in arr_split.values()),
            "n_test_arrangements": sum(v is Split.TEST for v in arr_split.values()),
            "n_dev_episodes": sum(s.split is Split.DEV for s in specs),
            "n_test_episodes": sum(s.split is Split.TEST for s in specs)}
