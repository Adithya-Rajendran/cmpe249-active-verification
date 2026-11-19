"""A navigation task with image observations, a stop action, physical contact and episode outcomes."""

from enum import Enum

from development.workshop_realism.robot_env import WorkshopEnv


class Outcome(str, Enum):
    RUNNING = "running"
    CROSSED = "crossed"
    SAFE_STOP = "safe_stop"
    HIT_DOOR = "hit_door"
    EARLY_STOP = "early_stop"
    TIMEOUT = "timeout"


class DoorApproachTask:
    """Drive through an open doorway, or stop safely before a closed one."""

    def __init__(self, env: WorkshopEnv, door_y: float):
        self.env = env
        self.door_y = door_y
        self.ended = True

    def reset(self, *, open_door, lateral=0.0, heading=0.0):
        self.open_door = bool(open_door)
        self.steps = 0
        self.ended = False
        return self.env.reset(
            target_present=False,
            door_fraction=float(self.open_door),
            lateral=lateral,
            heading=heading,
        )

    def step(self, forward_mps=0.0, turn_radps=0.0, *, stop=False):
        if self.ended:
            raise RuntimeError("Reset before the next episode")
        obs = self.env.step(0.0 if stop else forward_mps, 0.0 if stop else turn_radps)
        self.steps += 1
        trace = self.env.diagnostic()
        if trace["door_contacts"]:
            outcome = Outcome.HIT_DOOR
        elif trace["robot_xyz"][1] > self.door_y + 0.30:
            outcome = Outcome.CROSSED
        elif stop:
            outcome = Outcome.EARLY_STOP if self.open_door else Outcome.SAFE_STOP
        elif self.steps >= 120:
            outcome = Outcome.TIMEOUT
        else:
            outcome = Outcome.RUNNING
        self.ended = outcome != Outcome.RUNNING
        reward = (
            (1.0 if outcome in (Outcome.CROSSED, Outcome.SAFE_STOP) else -1.0)
            if self.ended
            else 0.0
        )
        return obs, reward, self.ended, {"outcome": outcome.value, "steps": self.steps}
