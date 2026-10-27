"""Render installed mount meshes and exact reference points; no robot/SDK access."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/autoweld-matplotlib')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/tcp-reference'
OUT.mkdir(exist_ok=True)
manifest = json.loads((ROOT / 'assets/camera_mount/manifest.json').read_text())
handeye = json.loads((ROOT / 'runtime/agents-live/handeye.json').read_text())
tool = np.array(manifest['flange_from_camera'], float)
camera = tool @ np.array(handeye['tcp_from_camera'], float)
points = np.array([np.zeros(3), tool[:3, 3]*1000, camera[:3, 3]*1000])
colors = ['#475569', '#1768bc', '#d86619']

fig = plt.figure(figsize=(15.5, 9), facecolor='white')
fig.text(.045, .945, 'Where weld_tcp is defined', fontsize=25, weight='bold', color='#152b3c')
fig.text(.045, .907, 'Installed receiver / camera mount CAD. Markers use exact model coordinates.',
         fontsize=12, color='#526676')
ax = fig.add_axes([.015, .14, .64, .73], projection='3d')
ax.set_proj_type('ortho')
for index, entry in enumerate(manifest['meshes']):
    mesh = trimesh.load(ROOT / 'assets/camera_mount' / entry['file'], force='mesh', process=False)
    # Convex hulls keep the overview legible and small; all geometry remains in
    # the source mesh frame. No additional kinematic model is instantiated.
    mesh = mesh.convex_hull
    vertices = mesh.vertices @ Rotation.from_euler('xyz', entry['rpy']).as_matrix().T
    vertices += np.array(entry['xyz'])*1000
    faces = vertices[mesh.faces]
    normal = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    shade = .55 + .4*np.abs(normal @ np.array([.3, -.5, .8124]))
    base = np.array([.56, .62, .67]) if index < 2 else np.array([.23, .28, .31])
    facecolors = np.c_[np.clip(shade[:, None]*base, 0, 1), np.full(len(faces), .68 if index < 2 else .82)]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolors, edgecolors='none', zsort='average'))

ax.set(xlim=(-112, 38), ylim=(-42, 108), zlim=(-130, 20))
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=-127)
ax.set_axis_off()
ax.plot(*np.vstack([points[1], points[2]]).T, '--', color='#79685b', lw=2)
for i, (point, color) in enumerate(zip(points, colors)):
    # Draw all markers after the geometry in projection: these are annotated
    # reference points, including the tool origin inside the mounting interface.
    xyz = proj3d.proj_transform(*point, ax.get_proj())
    ax.annotate('ABC'[i], xy=xyz[:2], xytext=(0, 0), textcoords='offset points',
                color='white', fontsize=12, weight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=.35', fc=color, ec='white', lw=1.8), zorder=100)

for direction, name in zip(np.eye(3)*19, ('+X', '+Y', '+Z')):
    a = proj3d.proj_transform(*points[0], ax.get_proj())
    b = proj3d.proj_transform(*direction, ax.get_proj())
    ax.annotate('', xy=b[:2], xytext=a[:2], arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2), zorder=90)
    ax.annotate(name, xy=b[:2], color='#526676', fontsize=9, xytext=(3, 3), textcoords='offset points', zorder=100)

optical_end = points[2] + camera[:3, 2]*28
a = proj3d.proj_transform(*points[2], ax.get_proj())
b = proj3d.proj_transform(*optical_end, ax.get_proj())
ax.annotate('', xy=b[:2], xytext=a[:2], arrowprops=dict(arrowstyle='->', color=colors[2], lw=2), zorder=90)
ax.annotate('camera looks this way', xy=b[:2], xytext=(-45, -16), textcoords='offset points',
            color=colors[2], fontsize=10, zorder=100)

labels = [
    ('A', 'J6 tool-frame origin', 'At the receiver mounting interface in the model.', '0, 0, 0 mm'),
    ('B', 'weld_tcp — model reference', 'Virtual point specified by the mount CAD.', '−69.50, +32.10, −68.78 mm'),
    ('C', 'Calibrated camera origin', 'Estimated optical origin from hand–eye calibration.', '−46.85, +58.00, −56.53 mm'),
]
for i, (letter, title, detail, coordinate) in enumerate(labels):
    y = .80-i*.165
    fig.text(.675, y, letter, fontsize=15, weight='bold', color=colors[i],
             bbox=dict(boxstyle='circle,pad=.3', fc='#f3f6f8', ec=colors[i], lw=1.5))
    fig.text(.714, y+.008, title, fontsize=13, weight='bold', color=colors[i])
    fig.text(.675, y-.045, detail, fontsize=10.5, color='#44596a')
    fig.text(.675, y-.08, coordinate, fontsize=12, family='monospace', color='#152b3c')
fig.text(.675, .265, 'All coordinates above are in A’s axes.', fontsize=11, color='#44596a')
fig.text(.675, .225, 'B → C: 36.53 mm', fontsize=16, weight='bold', color='#152b3c')
fig.text(.675, .185, 'A → C: 93.57 mm', fontsize=13, color='#44596a')
fig.text(.045, .10, 'B is a mathematical reference, not a visible spot to touch with a ruler.',
         fontsize=12, weight='bold', color='#152b3c')
fig.text(.045, .063, 'Model geometry shown with simplified surfaces; markers may lie inside the assembly. This is not a photograph.',
         fontsize=10, color='#526676')
fig.savefig(OUT / 'weld-tcp-marked.png', dpi=140, facecolor='white')
fig.savefig(OUT / 'weld-tcp-marked.svg', facecolor='white')
(OUT / 'reference-points.json').write_text(json.dumps(dict(
    coordinate_frame='J6 gripper/tool frame', units='mm', tool_from_nominal_tcp=tool.tolist(),
    tool_from_calibrated_camera=camera.tolist(), points_mm=points.tolist(),
    nominal_tcp_to_camera_mm=float(np.linalg.norm(points[2]-points[1])),
    tool_origin_to_camera_mm=float(np.linalg.norm(points[2]))), indent=2)+'\n')
print(OUT / 'weld-tcp-marked.png')
