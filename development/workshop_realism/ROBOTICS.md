# Use the workshop for robot training

The [MuJoCo training archive](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/workshop-mujoco-training.zip) contains eight native models, their textures, a mobile robot and a fitted example policy.
The [Isaac Sim package](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/workshop-isaac-facilities.zip) contains textured facility geometry, collision shapes, and a driven sliding door. Add your own robot to the Isaac stage.

## Run a MuJoCo episode

Extract `workshop-mujoco-training.zip`. Open the extracted directory.
Use Python with MuJoCo 3.17.0, NumPy and Pillow. Configure a supported OpenGL backend for RGB observations.

```python
from development.workshop_realism.robot_env import WorkshopEnv
from development.workshop_realism.training_task import DoorApproachTask

env = WorkshopEnv.from_mjcf("layouts/COR01.xml")
try:
    task = DoorApproachTask(env, float(env.model.body("blocker").pos[1]))
    observation = task.reset(open_door=True)
    finished = False
    while not finished:
        observation, reward, finished, info = task.step(0.34, 0.0)
    print(info["outcome"])
finally:
    env.close()
```

The drive command accepts forward speed (m/s) and turn rate (rad/s).
Use the stop action to end an episode. Each control step advances 48 ms.
Observations contain mast RGB and two wheel encoder readings. Use `env.frame("robot_low")` for the lower camera.
The task rewards crossing an open doorway or stopping before a closed door.

Use `env.reset(target_present=True, door_fraction=0.5)` to place the target and start the door halfway open.
Use the facility input on `env.step` to drive the door during an episode. The door servo limits speed and force.
Use `env.diagnostic()` for physics checks. Keep this privileged state outside a camera policy.

Load the example policy and test every packaged model from the extracted directory.

```sh
python3 -m development.workshop_realism.portable_robot_check .
```

Fit a new policy on the six training layouts. Evaluate it on the two reserved layouts.

```sh
python3 -m development.workshop_realism.train_navigation --layouts layouts --out new-policy-run
```

Test the supplied policy at five door positions. Each start also has a test with a straight drive command.

```sh
python3 -m development.workshop_realism.partial_openings . new-clearance-results.json
```

The example results record the actual contact and crossing outcomes. Open `example-policy/frames` for the input camera images.

## Open the Isaac Sim facility

Open a supplied `facility.usdz` stage. The stage contains the door joint at `/World/DoorSlide`.
Set the linear drive target within its joint limits. The parking side determines the sign.
The original target and furnished workshop are present in the exported scene. The guided test cart exists only during acceptance runs.

The MuJoCo task uses a differential-drive base with two powered wheels and two free ball casters.
The Isaac acceptance test uses a guided cart to check contact and passage. The native physics reports record these distinct tests.
The navigation demo trains a camera policy to stop or drive through an entrance. It has its own layout split and outcomes.

Read the asset license files before redistributing the models. The package includes the YCB attribution and texture credits.
