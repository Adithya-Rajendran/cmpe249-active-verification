"""Arrangement manifest -> episode specs + scoring-only ground truth (see verify/README.md section 4).

The split is by arrangement only: every episode of an arrangement (all conditions, all lightings)
lands in the same split. Ground truth is returned separately and never goes to ReplayEnv or a policy.

Checkpoint 2 additions (docs/SIM_PREREGISTRATION.md sections 2, 3 and 8):
  - ArrangementRecord carries the optional quota-slot fields occluder_family (S8, M16, IND28, white_box, ycb_box),
    m16_side ("L" / "R", M16 only) and design_case (bool), written by sim/episodes.py as plain manifest keys.
    bootstrap_strata() turns them into the stratum of the family-stratified cluster bootstrap: the family, with
    M16 split by side, and design cases kept apart from the random slots.
  - A top-level "simulation": true marks a manifest of Isaac Sim renders (load_manifest_info reads it); verify.run
    and verify.select then label every output SIMULATION.
  - load_geometric_labels reads the sim view labels (cmpe249.sim-view-labels-set.v0): per-view geometric diagnostic
    labels, for development analysis and scoring only, never a policy input.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verify.schema import (
    CONDITIONS,
    Condition,
    EpisodeSpec,
    GroundTruth,
    Lighting,
    ManifestError,
    Split,
    episode_id,
    split_arrangements,
    truth_from_condition,
)

MANIFEST_SCHEMA = "cmpe249.arrangements.v1"
SIM_VIEW_LABELS_SCHEMA = "cmpe249.sim-view-labels-set.v0"   # written by sim/episodes.py (view_labels.json)
# Quota-slot classes (docs/SIM_PREREGISTRATION.md, occluder quotas). M16 is split by side in the strata.
OCCLUDER_FAMILIES = ("S8", "M16", "IND28", "white_box", "ycb_box")
M16_SIDES = ("L", "R")
SIMULATION_LABEL = "SIMULATION - Isaac Sim renders; every result is a simulation result"
UNSTRATIFIED_NOTE = ("no occluder-family information for every arrangement: the cluster bootstrap is UNSTRATIFIED "
                     "(it resamples arrangements over the whole set, so a replicate need not keep the design's quota "
                     "counts)")
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
    occluder_family: str | None = None   # quota slot family; None for legacy (pilot) manifests
    m16_side: str | None = None          # "L" / "R" for M16, None otherwise
    design_case: bool = False            # DES0..DES3: dev checks outside the random quota slots

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", Split(self.split))
        if self.occluder_family is not None and self.occluder_family not in OCCLUDER_FAMILIES:
            raise ManifestError(f"{self.arrangement_id}: occluder_family must be one of {OCCLUDER_FAMILIES} or null, "
                                f"got {self.occluder_family!r}")
        if self.occluder_family == "M16":
            if self.m16_side not in M16_SIDES:
                raise ManifestError(f"{self.arrangement_id}: an M16 arrangement needs m16_side 'L' or 'R' (its quota "
                                    f"slot), got {self.m16_side!r}")
        elif self.m16_side is not None:
            raise ManifestError(f"{self.arrangement_id}: m16_side is only for the M16 family, got {self.m16_side!r} "
                                f"with occluder_family {self.occluder_family!r}")
        if not isinstance(self.design_case, bool) or not isinstance(self.indistinguishable, bool):
            raise ManifestError(f"{self.arrangement_id}: design_case and indistinguishable must be booleans")
        lights = tuple(Lighting(l) for l in self.lightings)
        if not lights or len(set(lights)) != len(lights):
            raise ManifestError(f"{self.arrangement_id}: lightings must be non-empty and unique")
        object.__setattr__(self, "lightings", tuple(sorted(lights, key=_LIGHT_ORDER.__getitem__)))
        if not RE_ARRANGEMENT.match(self.arrangement_id):
            raise ManifestError(f"bad arrangement id {self.arrangement_id!r} (expected e.g. A01)")

    @property
    def has_slot(self) -> bool:
        """True when the record carries its quota-slot fields (occluder family)."""
        return self.occluder_family is not None

    @property
    def stratum(self) -> str | None:
        """The bootstrap stratum (quota slot class): the family, M16 split by side ("M16-L" / "M16-R"); design
        cases get their own "design:" strata so they never mix with the random slots. None without a family."""
        if self.occluder_family is None:
            return None
        base = f"M16-{self.m16_side}" if self.occluder_family == "M16" else self.occluder_family
        return f"design:{base}" if self.design_case else base

    def to_dict(self) -> dict:
        d = {"arrangement_id": self.arrangement_id, "target_id": self.target_id,
             "split": self.split.value, "lightings": [l.value for l in self.lightings],
             "indistinguishable": self.indistinguishable}
        if self.occluder_family is not None or self.design_case:   # legacy records keep their exact shape
            d.update(occluder_family=self.occluder_family, m16_side=self.m16_side, design_case=self.design_case)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ArrangementRecord:
        ind, des = d.get("indistinguishable", False), d.get("design_case", False)
        if not isinstance(ind, bool) or not isinstance(des, bool):   # "false" (a string) would read as True
            raise ManifestError(f"{d.get('arrangement_id')}: indistinguishable and design_case must be JSON booleans")
        return cls(d["arrangement_id"], d["target_id"], Split(d["split"]),
                   tuple(Lighting(l) for l in d.get("lightings", ("L1", "L2"))),
                   ind, d.get("occluder_family"), d.get("m16_side"), des)


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


def save_manifest(arrs: Sequence[ArrangementRecord], path: str | Path, *, simulation: bool | None = None) -> None:
    """simulation: None writes no flag (legacy shape); True / False writes the top-level "simulation" key."""
    episodes_from_arrangements(arrs)  # validates before writing
    doc: dict[str, Any] = {"schema": MANIFEST_SCHEMA}
    if simulation is not None:
        doc["simulation"] = bool(simulation)
    doc["arrangements"] = [a.to_dict() for a in arrs]
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


@dataclass(frozen=True)
class ManifestInfo:
    """A manifest file's arrangements and its top-level flags."""
    arrangements: tuple[ArrangementRecord, ...]
    simulation: bool          # top-level "simulation": true (sim/episodes.py); False when absent

    def strata(self) -> dict[str, str] | None:
        return bootstrap_strata(self.arrangements)


def _read_manifest(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: schema is {doc.get('schema')!r}, expected {MANIFEST_SCHEMA!r}")
    sim = doc.get("simulation", False)
    if not isinstance(sim, bool):   # "true" as a string must not pass for the flag (nor "false" for its absence)
        raise ManifestError(f"{path}: top-level 'simulation' must be a JSON boolean, got {sim!r}")
    return doc


def load_manifest_info(path: str | Path) -> ManifestInfo:
    doc = _read_manifest(path)
    arrs = tuple(ArrangementRecord.from_dict(d) for d in doc["arrangements"])
    return ManifestInfo(arrs, bool(doc.get("simulation", False)))


def load_arrangements(path: str | Path) -> list[ArrangementRecord]:
    return list(load_manifest_info(path).arrangements)


def load_manifest(path: str | Path) -> tuple[list[EpisodeSpec], GroundTruth]:
    return episodes_from_arrangements(load_arrangements(path))


def bootstrap_strata(arrs: Iterable[ArrangementRecord]) -> dict[str, str] | None:
    """{arrangement_id: stratum} for the family-stratified cluster bootstrap, or None when any arrangement lacks
    its occluder family (the caller then falls back to the unstratified bootstrap and flags it, UNSTRATIFIED_NOTE).
    A partly annotated manifest also gives None: a stratum built from some arrangements only would not keep the
    design's quota counts."""
    arrs = list(arrs)
    if not arrs or any(not a.has_slot for a in arrs):
        return None
    return {a.arrangement_id: a.stratum for a in arrs}   # type: ignore[misc]


# ---------------------------------------------------------------- geometric view labels (sim)

@dataclass(frozen=True)
class GeometricLabels:
    """Per-view geometric diagnostic labels of a sim set (sim/episodes.py view_labels.json). Development analysis and
    scoring only: never a policy input. episodes: episode_id -> per-stop diagnostic flags, stop order 0..n-1."""
    episodes: Mapping[str, tuple[bool, ...]]
    arrangement_of: Mapping[str, str]
    lighting_of: Mapping[str, Lighting]
    condition_of: Mapping[str, Condition]
    truth_of: Mapping[str, str]
    rules: Mapping[str, Any]
    simulation: bool

    def no_diagnostic_view(self) -> dict[str, bool]:
        """episode_id -> True when no stop of the episode is geometrically diagnostic."""
        return {e: not any(d) for e, d in self.episodes.items()}

    def restricted(self, episode_ids: Iterable[str]) -> GeometricLabels:
        keep = set(episode_ids)
        sel = lambda m: {e: v for e, v in m.items() if e in keep}   # noqa: E731
        return GeometricLabels(sel(self.episodes), sel(self.arrangement_of), sel(self.lighting_of),
                               sel(self.condition_of), sel(self.truth_of), self.rules, self.simulation)


def load_geometric_labels(path: str | Path, n_stops: int | None = None) -> GeometricLabels:
    """Read a cmpe249.sim-view-labels-set.v0 file. Every episode must name its arrangement, condition and lighting
    consistently with its episode_id, list its stops 0..n-1 exactly once (n_stops when given) and give a boolean
    "diagnostic" for each; anything else raises ManifestError."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SIM_VIEW_LABELS_SCHEMA:
        raise ManifestError(f"{path}: view labels schema must be {SIM_VIEW_LABELS_SCHEMA!r}, got {doc.get('schema')!r}")
    eps: dict[str, tuple[bool, ...]] = {}
    arr_of: dict[str, str] = {}
    light_of: dict[str, Lighting] = {}
    cond_of: dict[str, Condition] = {}
    truth_of: dict[str, str] = {}
    for eid, e in sorted(doc["episodes"].items()):
        cond, light, arr = Condition(e["condition"]), Lighting(e["lighting"]), e["arrangement_id"]
        if episode_id(arr, cond, light) != eid:
            raise ManifestError(f"{path}: episode {eid!r} says arrangement {arr!r}, condition {cond.value!r}, "
                                f"lighting {light.value!r}")
        stops = sorted(e["stops"], key=lambda s: int(s["stop"]))
        idx = [int(s["stop"]) for s in stops]
        want = n_stops if n_stops is not None else len(stops)
        if idx != list(range(want)):
            raise ManifestError(f"{path}: {eid} lists stops {idx}, expected each of 0..{want - 1} once")
        diag = tuple(s["diagnostic"] for s in stops)
        if not all(isinstance(d, bool) for d in diag):
            raise ManifestError(f"{path}: {eid}: every 'diagnostic' must be a JSON boolean")
        eps[eid], arr_of[eid], light_of[eid], cond_of[eid] = diag, arr, light, cond
        truth_of[eid] = str(e.get("truth", truth_from_condition(cond).value))
        if truth_of[eid] != truth_from_condition(cond).value:
            raise ManifestError(f"{path}: {eid} has truth {truth_of[eid]!r}, its condition says "
                                f"{truth_from_condition(cond).value!r}")
    return GeometricLabels(eps, arr_of, light_of, cond_of, truth_of, dict(doc.get("rules", {})),
                           bool(doc.get("simulation", False)))


def no_diagnostic_share(no_diag: Mapping[str, bool], episode_ids: Iterable[str]) -> tuple[int, int]:
    """(episodes with no geometrically diagnostic stop, episodes) over the given ids; KeyError for an id without
    labels."""
    ids = list(episode_ids)
    return sum(bool(no_diag[e]) for e in ids), len(ids)


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
