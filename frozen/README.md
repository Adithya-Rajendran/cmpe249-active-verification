# Frozen inputs of the test evaluation

Byte-identical copies of development outputs that the test path reads. They are frozen by the commit of docs/FREEZE.md
that records their sha256, made after the researcher's final review, and they change only by a dated addendum.
Provenance: `notes/2026-11-06-pooled-calibration.md`. The candidate's own `status` string predates the freeze, and
FREEZE.md supersedes it.

| File | sha256 | Source |
|---|---|---|
| `frozen_candidate.json` | 3a45b23cbe82856660a9c1e34de3074238b8d8204512663d972a215939e9c26d | `python -m verify.select` (select-v4 revision 2) on both environments' 120 development arrangements, with `--ranker A --ranker A-nocue`, at commit 78263bc; eligible, complete grouped CV. Thresholds tau_q 0.3, tau_occ 0.4, tau_match 0.7, k 2, lam 0. |
| `frozen_candidate.json.sha256` | (sidecar) | Written with the candidate; `verify.select.require_freeze_eligible` checks the candidate against it. |
| `model_A.json` | 297e11af452d169525dd60c974291e5969e7c124194a809398f7519663381a19 | `python -m verify.ranker --name A` on the same 120 arrangements, with this candidate's thresholds; C 10 chosen by the pinned inner arrangement CV. |
| `model_A-nocue.json` | eab90ded6364d689d59104070b52fd47d851818ca9aebd1d718ee3c8c93370c8 | `python -m verify.ranker --name A-nocue`, the same way; C 1. |

`verify.run --candidate frozen/frozen_candidate.json --model A=frozen/model_A.json --model A-nocue=frozen/model_A-nocue.json`
refuses a model whose recorded features, thresholds or candidate sha256 differ from these.
