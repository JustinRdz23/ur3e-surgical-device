#!/usr/bin/env python3
"""Procedural curved suture needle: one source of truth for MuJoCo and Unity.

Generates:
  mesh/high_res/needle.obj      visual mesh in MuJoCo frame (Z-up, metres), next
                                to the tool meshes so it shares their meshdir
  <unity>/Assets/Models/Needle/needle.obj
                                same file for Unity; MujocoBridge.cs rotates it
                                like the tool meshes (MuJoCo (x,y,z) -> Unity (x,z,y))
  mjcf/needle/needle_assets.xml mesh, material and geom classes; include at top level
  mjcf/needle/needle_body.xml   needle body at the origin; include inside a <frame>
                                to place it in a scene
  mjcf/needle_drop.xml          table + needle dropped from a few cm, for mujoco.viewer

Needle frame (body "needle"):
  origin  = centre of the circle the needle is bent on
  arc     = lies in the local XY plane, from angle 0 (tail / swage) to
            ARC_FRACTION * 2*pi (tip), turning counter-clockwise about +Z
"""

from pathlib import Path

import numpy as np

# ---- needle geometry (SI units) -------------------------------------------
RADIUS = 0.008            # radius of curvature [m]
ARC_FRACTION = 3.0 / 8.0  # 3/8 circle is the most common suture needle
WIRE_DIAMETER = 0.0006    # [m]
TIP_TAPER = 0.15          # fraction of the arc length that tapers to the point
DENSITY = 7850.0          # stainless steel [kg/m^3]

# ---- discretisation --------------------------------------------------------
N_PATH = 120   # rings along the arc (visual mesh)
N_RING = 16    # vertices per ring (visual mesh)
N_CAPSULES = 12  # collision capsules along the arc

# ---- scene -----------------------------------------------------------------
DROP_HEIGHT = 0.03  # [m] above the table top

NEEDLE_DIR = Path(__file__).resolve().parent
MJCF_DIR = NEEDLE_DIR.parent
MESH_DIR = MJCF_DIR.parent / "mesh" / "high_res"
REPO_ROOT = MJCF_DIR.parents[2]
UNITY_MESH_DIR = REPO_ROOT / "unity-sim" / "UR3-medical-robot" / "Assets" / "Models" / "Needle"


def arc_angle():
    return ARC_FRACTION * 2.0 * np.pi


def wire_radius_at(s):
    """Wire radius at normalised arc position s in [0, 1] (0 = tail, 1 = tip)."""
    r = WIRE_DIAMETER / 2.0
    s0 = 1.0 - TIP_TAPER
    if s <= s0:
        return r
    # Smooth (cosine) taper so the point does not look like a cone glued on.
    t = (s - s0) / TIP_TAPER
    return r * 0.5 * (1.0 + np.cos(np.pi * t))


def centreline(s):
    """Point and unit tangent of the arc centreline at s in [0, 1]."""
    a = s * arc_angle()
    p = np.array([RADIUS * np.cos(a), RADIUS * np.sin(a), 0.0])
    t = np.array([-np.sin(a), np.cos(a), 0.0])
    return p, t


# ---- visual mesh -----------------------------------------------------------
def build_mesh():
    """Tube swept along the arc; tail capped with a fan, tip closed to a point."""
    verts, faces = [], []
    z = np.array([0.0, 0.0, 1.0])

    for i in range(N_PATH):
        s = i / (N_PATH - 1)
        p, t = centreline(s)
        n = np.cross(t, z)  # radial direction (outward); with z spans the cross-section
        r = wire_radius_at(s)
        for j in range(N_RING):
            phi = 2.0 * np.pi * j / N_RING
            verts.append(p + r * (np.cos(phi) * n + np.sin(phi) * z))

    # Side quads, split in two triangles, wound so normals point outwards.
    for i in range(N_PATH - 1):
        for j in range(N_RING):
            a = i * N_RING + j
            b = i * N_RING + (j + 1) % N_RING
            c = (i + 1) * N_RING + (j + 1) % N_RING
            d = (i + 1) * N_RING + j
            faces.append((a, c, b))
            faces.append((a, d, c))

    # Tail cap: fan around the tail centre, facing -tangent.
    tail_centre = len(verts)
    verts.append(centreline(0.0)[0])
    for j in range(N_RING):
        faces.append((tail_centre, j, (j + 1) % N_RING))

    # Tip: last ring has radius 0, so it is already a degenerate point; collapse
    # it to a single vertex to keep the mesh watertight without slivers.
    tip_ring = (N_PATH - 1) * N_RING
    tip = len(verts)
    verts.append(centreline(1.0)[0])
    faces = [tuple(tip if v >= tip_ring and v < tip_ring + N_RING else v for v in f)
             for f in faces]
    faces = [f for f in faces if len(set(f)) == 3]

    return np.array(verts), np.array(faces)


def write_obj(path, verts, faces, header):
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        for v in verts:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ---- collision capsules ----------------------------------------------------
def capsule_segments():
    """Chords of the arc. MuJoCo would convex-hull a mesh (turning the arc into
    a solid 'D'), so collision is a chain of capsules that follows the curve."""
    segs = []
    for k in range(N_CAPSULES):
        s0, s1 = k / N_CAPSULES, (k + 1) / N_CAPSULES
        p0, _ = centreline(s0)
        p1, _ = centreline(s1)
        # Capsule radius = the thinnest point of this segment, so collision never
        # sticks out of the visual; floor it so the tip still has some contact.
        r = max(min(wire_radius_at(s0), wire_radius_at(s1)), 0.25 * WIRE_DIAMETER / 2.0)
        segs.append((p0, p1, r))
    return segs


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


HEADER = "<!-- Generated by needle_gen.py — edit the script, not this file. -->\n"


def write_needle_assets(path):
    path.write_text(HEADER + f"""<mujoco>
  <asset>
    <mesh name="needle_mesh" file="needle.obj"/>
    <material name="needle_steel" rgba="0.78 0.80 0.83 1" specular="0.9" shininess="0.9" reflectance="0.2"/>
  </asset>

  <default>
    <default class="needle_col">
      <!-- Collision only: invisible (group 3), carries all the mass.
           solref/solimp scaled to a sub-mm object: MuJoCo's defaults (20 ms
           time constant, 1 mm impedance width) let it sink millimetres.
           solref time constant must stay >= 2 * timestep.
           MuJoCo averages solref/solimp of both geoms in a contact, weighted
           by solmix; the high solmix makes the needle's values win. -->
      <geom type="capsule" group="3" density="{DENSITY:g}" rgba="1 0.3 0.1 0.5"
            condim="4" friction="0.8 0.02 0.0001"
            solref="0.001 1" solimp="0.99 0.999 0.0001" solmix="1000"/>
    </default>
    <default class="needle_vis">
      <!-- Visual only: no collisions, no mass. -->
      <geom type="mesh" group="1" contype="0" conaffinity="0" density="0"/>
    </default>
  </default>
</mujoco>
""")


def write_needle_body(path):
    caps = "\n".join(
        f'  <geom name="needle_col_{k}" class="needle_col" fromto="{fmt(p0)} {fmt(p1)}" size="{r:.6g}"/>'
        for k, (p0, p1, r) in enumerate(capsule_segments())
    )
    path.write_text(HEADER + f"""<!-- Needs needle_assets.xml included at the top level of the scene. -->
<mujoco>
<body name="needle">
  <freejoint name="needle_free"/>
  <geom name="needle_visual" class="needle_vis" mesh="needle_mesh" material="needle_steel"/>
{caps}
  <site name="needle_tip" pos="{fmt(centreline(1.0)[0])}" size="0.0003" rgba="1 0 0 1"/>
  <site name="needle_tail" pos="{fmt(centreline(0.0)[0])}" size="0.0003" rgba="0 0 1 1"/>
</body>
</mujoco>
""")


def write_drop_scene(path):
    path.write_text(HEADER + f"""<mujoco model="needle_drop">
  <compiler angle="degree" meshdir="../mesh/high_res"/>

  <!-- Small, light object: short timestep, elliptic cones for stable friction. -->
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.001"/>
  </visual>

  <statistic center="0 0 0.01" extent="0.05"/>

  <include file="needle/needle_assets.xml"/>

  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="box" size="0.1 0.1 0.005" pos="0 0 -0.005" rgba="0.2 0.4 0.3 1"/>
    <frame pos="0 0 {DROP_HEIGHT:g}" euler="20 10 0">
      <include file="needle/needle_body.xml"/>
    </frame>
  </worldbody>
</mujoco>
""")


def main():
    verts, faces = build_mesh()
    header = "needle_gen.py — MuJoCo body frame (Z-up, right-handed), metres"
    outputs = [MESH_DIR / "needle.obj"]
    write_obj(outputs[0], verts, faces, header)
    if UNITY_MESH_DIR.parent.is_dir():
        UNITY_MESH_DIR.mkdir(exist_ok=True)
        outputs.append(UNITY_MESH_DIR / "needle.obj")
        write_obj(outputs[-1], verts, faces, header)
    for writer, path in [(write_needle_assets, NEEDLE_DIR / "needle_assets.xml"),
                         (write_needle_body, NEEDLE_DIR / "needle_body.xml"),
                         (write_drop_scene, MJCF_DIR / "needle_drop.xml")]:
        writer(path)
        outputs.append(path)

    arc_len = RADIUS * arc_angle()
    print(f"needle: R={RADIUS*1e3:.1f} mm, {ARC_FRACTION:.3f} circle, "
          f"wire {WIRE_DIAMETER*1e3:.2f} mm, arc length {arc_len*1e3:.1f} mm")
    print(f"mesh:   {len(verts)} vertices, {len(faces)} triangles")
    for path in outputs:
        print(f"wrote:  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
