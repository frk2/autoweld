"""Place orientation trials around existing seed poses; no robot I/O."""

import json
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner


def main():
    root = Path("assets/stage3")
    model = tool_model()
    lines, arcs = [], []
    try:
        planner = MotionPlanner(model)
        for i, entry in enumerate(json.loads((root / "manifest.json").read_text())):
            T = planner.fk(np.array(entry["seed_q"]))
            p, R = T[:3, 3], T[:3, :3]
            direction = np.cos(i * 0.5) * R[:, 0] + np.sin(i * 0.5) * R[:, 1]
            line = cq.Edge.makeLine(
                cq.Vector(*((p - 0.01 * direction) * 1000)), cq.Vector(*((p + 0.01 * direction) * 1000))
            )
            radius = 0.0127
            normal, radial = R[:, 0], R[:, 2]
            center = p - radius * radial
            points = [
                cq.Vector(*((center + radius * Rotation.from_rotvec(normal * a).apply(radial)) * 1000))
                for a in [0, np.pi / 4, np.pi / 2]
            ]
            arc = cq.Edge.makeThreePointArc(*points)
            lines.append(line)
            arcs.append(arc)
            cq.exporters.export(line, str(root / f"perpendicular-line-{i + 1:02}.step"))
            cq.exporters.export(arc, str(root / f"center-arc-{i + 1:02}.step"))
        for name, edges in [("perpendicular-lines", lines), ("center-arcs", arcs)]:
            cq.exporters.export(cq.Compound.makeCompound(edges), str(root / f"{name}.step"))
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
