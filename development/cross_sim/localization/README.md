# Locating the remembered object before judging its state

The [protocol](PROTOCOL.md) fixes case selection, blind review and the decision rule.
The [results](../../../notes/2026-11-18-label-localization.md) include both instruction groups and their independent evidence review.

## Cohort

The cohort uses geometry to select four fresh qualification layouts.
The configuration fixes the seed blocks, target rules, camera motion, and ranking.
The builder checks earlier seeds, reservations, and layouts before generation.
It uses the project generator and the normal development seed gate.

The ranking uses projected hull coverage at each revisit camera.
Coverage above or equal to 0.90 is a screened geometry proxy.
This proxy does not certify that an image has zero visible target pixels.

Run the builder from the project root.

```sh
PYTHONPATH=. .venv/bin/python development/cross_sim/localization/cohort.py --out runs/2026-11-18-label-localization-v1/cohort
PYTHONPATH=. .venv/bin/python development/cross_sim/localization/validate_cohort.py
```

The validator checks candidate order and ranking against the generation log.
It regenerates selected layouts, coverage, and camera poses.
It checks the frame margins and historical layout uniqueness.

## Review procedure

1. Use `prepare.py` to select cases from both captures and create matched anonymous packets.
2. Run the `audit` mode in `reviews.py` before giving packets to readers.
3. Keep the `broker` mode active while two-step readers submit their first estimates.
4. Have the main reviewer record independent answers before opening either set of Terra answers.
5. Seal the three sets of readings with `reviews.py`.
6. Run `evidence.py` to measure each cited point and build the anonymous evidence sheets.
7. Review each explanation under its anonymous label and save the decisions.
8. Run `results.py` to reveal the groups and build the case table and illustrated casebook.
