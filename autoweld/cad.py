"""STEP tessellation through OpenCascade. STEP's native units are converted to mm by OCCT."""

from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh


def load_step(path):
    path = Path(path)
    if path.suffix.lower() not in (".step", ".stp"):
        raise ValueError("Select a STEP (.step or .stp) file")
    imported = cq.importers.importStep(str(path))
    meshes = []
    for shape in imported.vals():
        vertices, triangles = shape.tessellate(0.2, 0.1)
        if vertices and triangles:
            meshes.append(
                trimesh.Trimesh(
                    vertices=np.array([v.toTuple() for v in vertices]) * 0.001,
                    faces=np.asarray(triangles),
                    process=False,
                )
            )
    if not meshes:
        raise ValueError("STEP contains no tessellatable faces")
    return trimesh.util.concatenate(meshes)
