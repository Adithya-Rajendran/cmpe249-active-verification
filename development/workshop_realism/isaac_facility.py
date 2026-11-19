"""Export a collidable Isaac workshop and exercise the sliding joint with a guided test cart."""

import argparse
from copy import copy
import hashlib
import json
import traceback
from pathlib import Path


def run(args):
    import carb
    import omni.physx
    import omni.usd
    from pxr import Ar, Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from sim import isaac_batch as batch
    from development.workshop_realism.capture import isaac_stage

    arr = json.loads(args.scene.read_bytes())
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    built = isaac_stage(
        batch.build_stage, arr, args.models, args.tags, {}, args.textures
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    api.CreateTimeStepsPerSecondAttr(120)
    pos = arr["objects"]["occluder"]["poses"]
    x, y, z = pos["placed"]["base"]
    travel = pos["parked"]["base"][0]
    leaf = stage.GetPrimAtPath(built["occluder"])
    built["occluder_op"].Set(
        Gf.Matrix4d(*built["occluder_poses"]["placed"].reshape(-1).tolist())
    )

    def collider(prim):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateRestOffsetAttr(0.0)
        contact.CreateContactOffsetAttr(0.002)

    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Shelf")):
        if prim.IsA(UsdGeom.Cube):
            collider(prim)
    for prim in Usd.PrimRange(leaf):
        if prim.IsA(UsdGeom.Cube):
            collider(prim)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Objects")):
        if prim.IsA(UsdGeom.Mesh) and not str(prim.GetPath()).startswith(
            str(leaf.GetPath()) + "/"
        ):
            collider(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                "convexHull"
            )

    def rigid(prim, mass):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def slide(name, body, origin, axis, limits, stiffness, damping, force):
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/" + name)
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(force)
        return drive.CreateTargetPositionAttr(0.0)

    rigid(leaf, 42.0)
    drive = slide(
        "DoorSlide", leaf, (x, y, z), "X", sorted((0.0, travel)), 1250.0, 280.0, 230.0
    )
    stage_path = out / "facility.usdc"
    stage.GetRootLayer().Export(str(stage_path))
    packed = out / "facility.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(stage_path)), str(packed)):
        raise RuntimeError("Unable to package USD assets")
    reopened = Usd.Stage.Open(str(packed))
    packed_joint = UsdPhysics.PrismaticJoint.Get(reopened, "/World/DoorSlide")
    if (
        not packed_joint
        or abs(packed_joint.GetLowerLimitAttr().Get() - min(0.0, travel)) > 1e-5
        or abs(packed_joint.GetUpperLimitAttr().Get() - max(0.0, travel)) > 1e-5
    ):
        raise RuntimeError(
            "The packaged door joint did not reload with the specified limits"
        )
    from zipfile import ZipFile

    with ZipFile(packed) as package:
        images = [n for n in package.namelist() if n.lower().endswith(".png")]
        if package.testzip() is not None or len(images) < 13:
            raise RuntimeError("Packaged scene textures are incomplete")
    resolved_textures = set()
    for prim in reopened.Traverse():
        attribute = prim.GetAttribute("inputs:file")
        if not attribute:
            continue
        value = attribute.Get()
        if not isinstance(value, Sdf.AssetPath):
            raise RuntimeError(f"Unexpected texture input at {prim.GetPath()}")
        path = str(
            value.resolvedPath
            or Ar.GetResolver().Resolve(
                Sdf.ComputeAssetPathRelativeToLayer(reopened.GetRootLayer(), value.path)
            )
        )
        outer, inner = Ar.SplitPackageRelativePathOuter(path)
        if Path(outer).resolve() != packed.resolve() or inner not in images:
            raise RuntimeError(
                f"External or missing texture at {prim.GetPath()}: {path}"
            )
        resolved_textures.add(inner)
    if resolved_textures != set(images):
        raise RuntimeError("The reopened stage does not reference every packaged image")
    reopened_colliders = sum(
        prim.HasAPI(UsdPhysics.CollisionAPI) for prim in reopened.Traverse()
    )
    if reopened_colliders < 40:
        raise RuntimeError("Native scene lost its collision geometry")
    # The exported facility contains the original target and no diagnostic cart.
    target = stage.GetPrimAtPath(built["target"])
    target.SetActive(False)
    probe = UsdGeom.Xform.Define(stage, "/World/AcceptanceCart")
    initial = (0, y - 0.90, 0.16)
    probe.AddTranslateOp().Set(Gf.Vec3d(*initial))
    cube = UsdGeom.Cube.Define(stage, "/World/AcceptanceCart/Bumper")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(0.39, 0.33, 0.25))
    collider(cube.GetPrim())
    rigid(probe.GetPrim(), 10.0)
    cart = slide(
        "CartTestDrive", probe.GetPrim(), initial, "Y", [-0.05, 1.8], 350.0, 50.0, 55.0
    )
    settings = carb.settings.get_settings()
    settings.set_bool("/persistent/physics/updateToUsd", True)
    sim = omni.physx.get_physx_simulation_interface()
    interface = omni.physx.get_physx_interface()
    clock = 0
    dt = 1 / 120

    def pose(prim):
        if hasattr(interface, "get_rigidbody_transformation"):
            result = interface.get_rigidbody_transformation(str(prim.GetPath()))
            if result.get("ret_val"):
                return [float(v) for v in result["position"]]
        return list(
            UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
        )

    def advance(n):
        nonlocal clock
        for _ in range(n):
            sim.simulate(dt, clock * dt)
            sim.fetch_results()
            clock += 1

    sim.attach_stage(omni.usd.get_context().get_stage_id())
    records = []
    checks = {}
    try:
        for label, door_goal, cart_goal in [
            ("closed", 0.0, 0.0),
            ("blocked", 0.0, 1.45),
            ("back", 0.0, 0.0),
            ("middle", 0.5 * travel, 0.0),
            ("open", travel, 0.0),
            ("crossed", travel, 1.45),
        ]:
            drive.Set(door_goal)
            cart.Set(cart_goal)
            advance(540)
            actual = pose(leaf)
            vehicle = pose(probe.GetPrim())
            records.append(
                dict(
                    stage=label,
                    door_x=actual[0],
                    probe_y=vehicle[1],
                    door_goal_x=door_goal,
                )
            )
            checks[label + "_door"] = (
                abs(actual[0] - door_goal) < 0.012 and abs(actual[1] - y) < 0.004
            )
            if label == "blocked":
                checks["cart_blocks_at_closed_leaf"] = y - 0.25 < vehicle[1] < y - 0.12
            if label == "crossed":
                checks["cart_passes_open_leaf"] = vehicle[1] > y + 0.30
    finally:
        sim.detach_stage()
    data = {
        "layout": arr["base_id"],
        "pass": all(checks.values()),
        "checks": checks,
        "records": records,
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "packaged_textures": len(images),
        "resolved_texture_members": sorted(resolved_textures),
        "packaged_colliders": reopened_colliders,
        "reopened_joint_limits_m": [
            packed_joint.GetLowerLimitAttr().Get(),
            packed_joint.GetUpperLimitAttr().Get(),
        ],
        "physics_stage_sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        "usdz_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
        "scope": "Native Isaac constrained door with a guided contact cart. Differential-wheel rollouts run in MuJoCo.",
    }
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)
    if not data["pass"]:
        raise RuntimeError("Native USD door or cart check failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", type=Path)
    group.add_argument("--manifest", type=Path)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--tags", type=Path, required=True)
    p.add_argument("--textures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 360}
    )
    try:
        if args.scene is not None:
            run(args)
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            records = []
            for entry in json.loads(args.manifest.read_bytes())["arrangements"]:
                level = entry["levels"]["D0"]
                item = copy(args)
                item.scene = args.manifest.parent / level["path"]
                item.out = args.out / entry["base_id"]
                if (
                    hashlib.sha256(item.scene.read_bytes()).hexdigest()
                    != level["sha256"]
                ):
                    raise ValueError("Invalid scene binding")
                run(item)
                records.append(json.loads((item.out / "report.json").read_bytes()))
            (args.out / "summary.json").write_text(
                json.dumps(
                    {"passed": all(r["pass"] for r in records), "layouts": records},
                    indent=2,
                )
                + "\n"
            )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()
