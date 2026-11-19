# Reproduce the workshop comparison

Download the [workshop input archive](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/workshop-inputs.zip) to check the reported decisions.
It includes the scored images, native RGB and masks, and original model replies.

Run this command from the repository root. The pixel check used Python 3.10.12, Pillow 9.0.1 and NumPy 1.21.5.

```sh
python3 development/workshop_realism/supplement.py /path/to/workshop-inputs.zip
```

The checker reconstructs each delivered image from native RGB. It reads the simulator masks, recomputes each reference box, and compares the two physical states.
It checks the original paper settings for each verifier. It also compares each reported answer with the stored scorer output.

## Inspect the design

Generate the eight corrected workshop layouts in a new directory. The original generator used Python 3.13.5 and NumPy 2.5.3.

```sh
.venv/bin/python -m development.workshop_realism.door_clearance /tmp/cmpe249-workshop-scenes
```

Add `--fixture` to generate the separate engineering layout. Use a different output directory.
Use the [material inventory](materials/sources.json) to check the texture files.
See the [capture driver](capture.py) for the Isaac Sim and MuJoCo entry points.

The [protocol](PROTOCOL.md) defines the camera paths, the interventions and the decision rule.
The [door amendment](DOOR_AMENDMENT.md) defines the parking bay and the continuous clearance check.
Use the [robotics guide](ROBOTICS.md) to run the physical training task.

## Inspect the scores

Open `method-by-view.csv` for each camera and verifier. Open `paired-changes.csv` for matched comparisons with the previous corridor.
The columns for usable fraction record how much of the remembered box the geometry model could map.
Open `decisions.csv` for the individual answers and their image hashes.
[Study tables and figures](../../notes/figures/2026-11-18-workshop-clearance/)
