"""A differential-drive robot and an actuated sliding door in the furnished MuJoCo scene."""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WHEEL_RADIUS = 0.08
HALF_TRACK = 0.175
DT = 0.004
SUBSTEPS = 12


def training_xml(arr, models, tags, textures):
    from development.cross_sim import mujoco_scene
    from development.workshop_realism.capture import mj_scene
    from development.workshop_realism.door_clearance import TRAVEL, sweep

    if sweep(arr)["hits"]:
        raise ValueError("Door intersects static geometry")
    raw, _ = mj_scene(mujoco_scene.build_xml, arr, models, tags, textures)
    root = ET.fromstring(raw)
    root.set("model", "workshop_diff_drive")
    ET.SubElement(
        root,
        "option",
        timestep=str(DT),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        iterations="70",
    )
    root.find("compiler").set("angle", "radian")
    world = root.find("worldbody")
    act = ET.SubElement(root, "actuator")
    # Painted cladding and labels are visual. Solid architectural boxes supply contacts.
    for i, b in enumerate(arr["scene"]["boxes"]):
        g = world.find(f"geom[@name='box_{i}']")
        g.attrib.update(contype="1", conaffinity="1", friction="0.95 0.003 0.0001")
        g.set("name", b["name"])
    for body_name in ["target"] + [
        f"distractor_{i}" for i in range(len(arr["objects"]["distractors"]))
    ]:
        for g in world.find(f"body[@name='{body_name}']").findall("geom"):
            g.attrib.update(contype="1", conaffinity="1")
    door = world.find("body[@name='blocker']")
    poses = arr["objects"]["occluder"]["poses"]
    base = poses["placed"]["base"]
    sign = arr["design"]["door_mechanism"]["parking_side"]
    door.set("pos", " ".join(map(str, base)))
    door.set("gravcomp", "1")
    door.set("mocap", "false")
    ET.SubElement(
        door, "inertial", pos="0 0 1.02", mass="42", diaginertia="14.6 17.8 3.2"
    )
    ET.SubElement(
        door,
        "joint",
        name="door_slide",
        type="slide",
        axis=f"{sign} 0 0",
        range=f"0 {TRAVEL}",
        limited="true",
        damping="55",
        armature="1",
    )
    door.find("geom[@name='blocker_box']").attrib.update(contype="1", conaffinity="1")
    for g in door.findall("geom"):
        if g.get("type") == "box":
            g.attrib.update(contype="1", conaffinity="1")
    ET.SubElement(
        act,
        "position",
        name="door_servo",
        joint="door_slide",
        kp="1250",
        kv="280",
        ctrlrange=f"0 {TRAVEL}",
        forcerange="-230 230",
    )
    robot = ET.SubElement(
        world, "body", name="robot", pos=f"0 {base[1] - 0.90:.10f} .081"
    )
    ET.SubElement(robot, "freejoint", name="robot_free")
    ET.SubElement(
        robot, "inertial", pos="0 0 .062", mass="10", diaginertia=".13 .14 .20"
    )

    def geom(parent, name, **attrs):
        return ET.SubElement(
            parent, "geom", name=name, contype="1", conaffinity="1", **attrs
        )

    geom(
        robot,
        "bumper",
        type="box",
        size=".156 .165 .058",
        pos="0 0 .061",
        rgba=".2 .25 .27 1",
        friction=".7 .003 .0001",
    )
    for side, x in [("left", -HALF_TRACK), ("right", HALF_TRACK)]:
        wheel = ET.SubElement(robot, "body", name=side + "_wheel", pos=f"{x} 0 0")
        ET.SubElement(
            wheel,
            "joint",
            name=side + "_axle",
            type="hinge",
            axis="1 0 0",
            damping=".015",
            armature=".003",
        )
        geom(
            wheel,
            side + "_tire",
            type="cylinder",
            size=f"{WHEEL_RADIUS} .019",
            euler="0 1.5707963267948966 0",
            mass=".35",
            rgba=".08 .08 .08 1",
            friction="1.2 .004 .0001",
            condim="3",
        )
        ET.SubElement(
            act,
            "velocity",
            name=side + "_drive",
            joint=side + "_axle",
            kv="3",
            ctrllimited="true",
            ctrlrange="-12 12",
            forcerange="-3.2 3.2",
        )
    for i, y in enumerate([-0.126, 0.126]):
        caster = ET.SubElement(
            robot, "body", name=f"ball_caster_body_{i}", pos=f"0 {y} -.054"
        )
        ET.SubElement(
            caster,
            "joint",
            name=f"ball_caster_bearing_{i}",
            type="ball",
            damping=".00002",
        )
        geom(
            caster,
            f"robot_caster_{i}",
            type="sphere",
            size=".026",
            mass=".05",
            rgba=".25 .25 .25 1",
            friction=".9 .0001 .0001",
        )
    geom(
        robot,
        "sensor_mast",
        type="box",
        size=".018 .018 .421",
        pos="0 -.04 .443",
        rgba=".18 .21 .23 1",
    )
    for n, h, pitch in [("mast", 0.95, 13), ("low", 0.48, 8)]:
        a = math.radians(pitch)
        ET.SubElement(
            robot,
            "camera",
            name="robot_" + n,
            pos=f"0 .09 {h - 0.081}",
            xyaxes=f"1 0 0 0 {math.sin(a)} {math.cos(a)}",
            fovy=str(arr["scene"]["cameras"][3]["fovy_deg"]),
        )
    # A separate inspection camera shows the complete rail and both endpoints.
    ET.SubElement(
        world,
        "camera",
        name="door_inspection",
        pos=f"{sign * 0.46} -2.55 1.15",
        xyaxes="1 0 0 0 .04 .99919968",
        fovy="47",
    )
    return ET.tostring(root, encoding="unicode")


class WorkshopEnv:
    """Return wheel encoders and optional RGB. Set the facility door independently of locomotion."""

    def __init__(self, arr, models, tags, textures, *, rgb=True, imagesize=(640, 360)):
        self.xml = training_xml(arr, models, tags, textures)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self._start(rgb, imagesize)

    @classmethod
    def from_mjcf(cls, path, *, rgb=True, imagesize=(640, 360)):
        """Load a packaged model with its relative texture paths."""
        env = cls.__new__(cls)
        env.model = mujoco.MjModel.from_xml_path(str(path))
        env._start(rgb, imagesize)
        return env

    def _start(self, rgb, imagesize):
        self.data = mujoco.MjData(self.model)
        self.renderer = (
            mujoco.Renderer(self.model, height=imagesize[1], width=imagesize[0])
            if rgb
            else None
        )
        self.target = np.flatnonzero(
            self.model.geom_bodyid == self.model.body("target").id
        )
        self.door_body = self.model.body("blocker").id
        self.robot_body = self.model.body("robot").id
        self.door_q = int(self.model.joint("door_slide").qposadr[0])
        self.wheel_q = [
            int(self.model.joint(k + "_axle").qposadr[0]) for k in ("left", "right")
        ]
        self.act = [
            self.model.actuator(k).id
            for k in ("door_servo", "left_drive", "right_drive")
        ]
        self.target_color = self.model.geom_rgba[self.target].copy()
        self.target_group = self.model.geom_group[self.target].copy()
        self.initial_robot_y = float(self.model.body("robot").pos[1])
        self.door_travel = float(self.model.joint("door_slide").range[1])
        self.desired_door = 0.0
        self.reset()

    def reset(
        self, *, target_present=True, door_fraction=0.0, lateral=0.0, heading=0.0
    ):
        if not 0 <= door_fraction <= 1 or abs(lateral) > 0.30:
            raise ValueError("Reset pose outside entrance")
        mujoco.mj_resetData(self.model, self.data)
        self.desired_door = door_fraction * self.door_travel
        self.data.qpos[self.door_q] = self.desired_door
        q = int(self.model.joint("robot_free").qposadr[0])
        self.data.qpos[q : q + 7] = [
            lateral,
            self.initial_robot_y,
            0.081,
            math.cos(heading / 2),
            0,
            0,
            math.sin(heading / 2),
        ]
        self.model.geom_rgba[self.target] = self.target_color
        self.model.geom_group[self.target] = self.target_group
        if not target_present:
            self.model.geom_rgba[self.target, 3] = 0
            self.model.geom_group[self.target] = 5
        self.model.geom_contype[self.target] = int(target_present)
        self.model.geom_conaffinity[self.target] = int(target_present)
        self.data.ctrl[self.act] = [self.desired_door, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(28):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def step(self, forward_mps, turn_radps, door_fraction=None, *, render=True):
        goal = (
            self.desired_door
            if door_fraction is None
            else float(door_fraction) * self.door_travel
        )
        if (
            not all(np.isfinite([forward_mps, turn_radps, goal]))
            or not 0 <= goal <= self.door_travel
        ):
            raise ValueError("Invalid action")
        forward = float(np.clip(forward_mps, -0.5, 0.5))
        turn = float(np.clip(turn_radps, -1.5, 1.5))
        left = -(forward - turn * HALF_TRACK) / WHEEL_RADIUS
        right = -(forward + turn * HALF_TRACK) / WHEEL_RADIUS
        for _ in range(SUBSTEPS):
            self.desired_door += float(
                np.clip(goal - self.desired_door, -0.36 * DT, 0.36 * DT)
            )
            self.data.ctrl[self.act] = [self.desired_door, left, right]
            mujoco.mj_step(self.model, self.data)
        return self.observe(render=render)

    def frame(self, camera="robot_mast"):
        if self.renderer is None:
            raise RuntimeError("RGB disabled")
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observe(self, render=True):
        obs = {"wheel_radians": self.data.qpos[self.wheel_q].copy()}
        if self.renderer is not None and render:
            obs["rgb"] = self.frame()
        return obs

    def diagnostic(self):
        contacts = []
        m = self.model
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if self.door_body in (ba, bb):
                contacts.append([m.geom(a).name, m.geom(b).name, float(c.dist)])
        return {
            "door_m": float(self.data.qpos[self.door_q]),
            "robot_xyz": self.data.xpos[self.robot_body].copy().tolist(),
            "robot_quat": self.data.xquat[self.robot_body].copy().tolist(),
            "door_contacts": contacts,
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
