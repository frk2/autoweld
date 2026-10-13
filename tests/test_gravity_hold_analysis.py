"""A gravity scale fit must cancel opposite approach biases, with correct sign."""

import json

import numpy as np

from scripts.analyze_mixed_gravity_holds import analyze


def test_known_gravity_error_separated_from_hysteresis_and_constant_bias(tmp_path):
    holds, rows = [], []
    kp = np.array([80, 96, 153, 30, 14.4, 62.0])
    t = 0
    for i, g5 in enumerate((-0.018, 0.018, -0.009, 0.009)):
        gravity = np.array([0, 0, 0, 1.4 + 0.05 * i, g5, 0])
        correction = np.array([0, 0, 0, 0.08 * gravity[3], -0.2 * g5 + 0.03, 0])
        for side in (-1, 1):
            target = np.zeros(6)
            # Deliberately large, symmetric direction dependence.
            error = correction / kp + side * np.array([0, 0, 0, 0.004, 0.009, 0])
            holds.append(dict(pose=f"pose{i}", side=side, target=target.tolist(), begin=t, end=t + 1))
            rows.append(
                dict(
                    timestamp=t + 0.5,
                    q=(target - error).tolist(),
                    pid_kp=kp.tolist(),
                    torque=dict(gravity_unscaled_nm=gravity.tolist()),
                )
            )
            t += 2
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(dict(name="synthetic", requested_factors=[1, 1, 1, 1.32, 1, 1], holds=holds)))
    path.with_name("calibration-control.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = analyze(path)
    assert abs(result["fits"][0]["fitted_factor_origin"] - 1.4) < 1e-12
    assert abs(result["fits"][1]["delta_factor_with_intercept"] + 0.2) < 1e-12
    assert abs(result["fits"][1]["constant_bias_nm"] - 0.03) < 1e-12
    for p in result["pairs"]:
        np.testing.assert_allclose(p["half_hysteresis_deg"][3:5], -np.rad2deg([0.004, 0.009]))
