# Positions drafts (test fixtures)

Byte-for-byte copies of `data/raw/2026-10-10_far-camera/positions_far_{A,B15,B20}_draft.json` (the 2026-10-10 far-camera
drafts: arc parameters, one single `tilt_deg`, `floor_coords_cm`). `data/raw/` is gitignored, so a clean checkout has no
copy; the tests that compare the simulated shelf's positions with option A (`tests/test_sim_arrange.py`) read these.
Do not edit them: they are the record of the drafts, and `configs/sim/positions_shelf_A.json` is checked against option A.
