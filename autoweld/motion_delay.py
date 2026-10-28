"""Empirical exposure/output-angle delay calibration from a static laser surface.

This is a calibration calculation, never a substitute for camera clock mapping.
During small forward/reverse optical-Z probes, the same static surface must give
the same laser depth at the same measured camera position. A timing error opens
a direction-dependent loop. Fit the signed history lookup offset that closes it,
then test cycles excluded from the fit and retain a profile confidence interval.
No commanded joint position or assumed acquisition/processing latency is used.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar


class DelayCalibrationRejected(ValueError):
    pass


@dataclass(frozen=True)
class DelayFit:
    delay_s: float
    uncertainty_s: float
    confidence_interval_s: tuple[float, float]
    residual_rms_m: float
    heldout_rms_m: float
    direction_bias_m: float
    samples: int
    cycles: int
    measured_range_m: float
    depth_range_m: float
    method: str = 'bidirectional_static_surface_depth'
    heldout_passed: bool = True

    def as_dict(self):
        return asdict(self)


def _curve_fit(position, depth, weights):
    center = float(np.median(position))
    scale = float(np.ptp(position))
    if scale < .003:
        raise DelayCalibrationRejected('Measured probe range is less than 3 mm')
    x = (position-center)/scale
    design = np.stack([np.ones_like(x), x, x*x, x*x*x], axis=1)
    current = weights.copy()
    for _ in range(5):
        root = np.sqrt(current)
        coefficients = np.linalg.lstsq(design*root[:, None], depth*root, rcond=None)[0]
        residual = depth-design@coefficients
        sigma = max(2e-5, 1.4826*float(np.median(abs(residual-np.median(residual)))))
        current = weights*np.minimum(1., 1.5*sigma/np.maximum(abs(residual), 1e-12))
    return coefficients, center, scale, residual, current


def _predict(model, position):
    coefficients, center, scale = model[:3]
    x = (position-center)/scale
    return np.polynomial.polynomial.polyval(x, coefficients)


def fit_motion_delay(frame_times, laser_depth_m, cycle_ids, output_times, camera_z_m, *,
                     weights=None, search_s=(-.150, .150), max_uncertainty_s=.002,
                     max_residual_m=.00025, max_direction_bias_m=.00015):
    """Fit ``q_history(exposure_time + delay_s)`` using measured probe motion.

    ``output_times/camera_z_m`` are derived from provisional measured-angle FK,
    which is explicitly unqualified for publishing world clouds. Timestamps must
    already share CLOCK_MONOTONIC. Frame times are hardware-mapped exposure
    midpoints; later decoding/transport receipt times are deliberately absent.
    """
    times, depth = np.asarray(frame_times, float), np.asarray(laser_depth_m, float)
    cycles = np.asarray(cycle_ids)
    ot, oz = np.asarray(output_times, float), np.asarray(camera_z_m, float)
    if (times.ndim != 1 or depth.shape != times.shape or cycles.shape != times.shape
            or ot.ndim != 1 or oz.shape != ot.shape or len(times) < 60 or len(ot) < 20
            or not np.isfinite(np.r_[times, depth, ot, oz]).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(ot) <= 0)
            or np.any(depth <= 0)):
        raise DelayCalibrationRejected('Need ordered finite exposure depths and measured camera outputs')
    low, high = map(float, search_s)
    if (not np.isfinite([low, high, max_uncertainty_s, max_residual_m, max_direction_bias_m]).all()
            or low >= high or min(max_uncertainty_s, max_residual_m, max_direction_bias_m) <= 0):
        raise DelayCalibrationRejected('Invalid delay qualification limits')
    weight = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    if weight.shape != times.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise DelayCalibrationRejected('Frame weights must be finite and positive')
    # Every trial uses the SAME frames and real measured brackets. No candidate
    # can appear better by discarding frames or extrapolating outside history.
    if times[0]+low < ot[0] or times[-1]+high > ot[-1]:
        raise DelayCalibrationRejected('Probe history does not bracket the full delay search')
    unique_cycles = np.unique(cycles)
    if len(unique_cycles) < 3 or np.ptp(depth) < .002:
        raise DelayCalibrationRejected('At least three observable probe cycles are required')
    velocity = np.gradient(oz, ot)

    def candidate(delay):
        position = np.interp(times+delay, ot, oz)
        speed = np.interp(times+delay, ot, velocity)
        forward, reverse = speed > .002, speed < -.002
        if min(np.count_nonzero(forward), np.count_nonzero(reverse)) < 15:
            raise DelayCalibrationRejected('Probe must contain measured motion in both directions')
        model = _curve_fit(position, depth, weight)
        residual, robust_weights = model[3:]
        return float(np.sum(robust_weights*residual**2)), position, speed, model

    grid = np.linspace(low, high, max(151, int(np.ceil((high-low)/.002))+1))
    costs = np.array([candidate(delay)[0] for delay in grid])
    best = int(np.argmin(costs))
    if best in (0, len(grid)-1):
        raise DelayCalibrationRejected('Delay estimate reached the search boundary')
    optimized = minimize_scalar(lambda delay: candidate(delay)[0],
        bounds=(grid[best-1], grid[best+1]), method='bounded', options={'xatol': 1e-7})
    delay = float(optimized.x)
    cost, position, speed, model = candidate(delay)
    residual, robust_weights = model[3:]
    rms = float(np.sqrt(np.average(residual**2, weights=robust_weights)))
    direction_bias = abs(float(np.median(residual[speed > .002])-np.median(residual[speed < -.002])))
    # The surface response must observe normal motion; a flat/tangential trace
    # cannot identify timing, however low its residual happens to be.
    gain = abs(float(model[0][1]/model[2]))
    if gain < .25 or gain > 4.:
        raise DelayCalibrationRejected('Laser depth does not observably track normal probe motion')
    heldout, independent_delays = [], []
    for cycle in unique_cycles:
        training = cycles != cycle
        if np.count_nonzero(training) < 40 or np.count_nonzero(~training) < 10:
            raise DelayCalibrationRejected('Probe cycles have insufficient independent samples')
        # Estimate timing without this cycle too. Holding out only the surface
        # curve would let the validation images influence their own timing fit.
        def training_cost(offset):
            sample_position = np.interp(times[training]+offset, ot, oz)
            fitted = _curve_fit(sample_position, depth[training], weight[training])
            return float(np.sum(fitted[4]*fitted[3]**2))

        training_costs = np.array([training_cost(x) for x in grid])
        index = int(np.argmin(training_costs))
        if index in (0, len(grid)-1):
            raise DelayCalibrationRejected('Held-out delay estimate reached the search boundary')
        independent = minimize_scalar(training_cost, bounds=(grid[index-1], grid[index+1]),
            method='bounded', options={'xatol': 1e-7})
        independent_delays.append(float(independent.x))
        independent_position = np.interp(times+independent.x, ot, oz)
        fitted = _curve_fit(independent_position[training], depth[training], weight[training])
        error = depth[~training]-_predict(fitted, independent_position[~training])
        heldout.append(float(np.sqrt(np.average(error**2, weights=weight[~training]))))
    heldout_rms = max(heldout)
    # One-parameter 95% profile interval, with a 20 um observation floor so
    # numerical/synthetic exactness never becomes a zero uncertainty claim.
    variance = max((2e-5)**2, cost/max(1., np.sum(robust_weights)-5))
    threshold = cost+3.841459*variance
    fine = np.unique(np.r_[np.linspace(low, high, int(np.ceil((high-low)/.0001))+1), delay])
    supported = fine[np.array([candidate(x)[0] <= threshold for x in fine])]
    if not len(supported) or supported[0] == low or supported[-1] == high:
        raise DelayCalibrationRejected('Delay confidence interval is unbounded in the search')
    lo, hi = float(min(supported[0], delay)), float(max(supported[-1], delay))
    # Correlated image/encoder errors do not average away as independent pixel
    # noise. Include cycle-level jackknife scatter and each held-out estimate.
    independent_delays = np.asarray(independent_delays)
    jackknife_se = np.sqrt((len(unique_cycles)-1)/len(unique_cycles)
        * np.sum((independent_delays-independent_delays.mean())**2))
    uncertainty = max(delay-lo, hi-delay, .00005, 1.96*float(jackknife_se),
        float(np.max(abs(independent_delays-delay))))
    if rms > max_residual_m or heldout_rms > max_residual_m or direction_bias > max_direction_bias_m:
        raise DelayCalibrationRejected('Probe failed static-surface held-out/direction consistency')
    if uncertainty > max_uncertainty_s:
        raise DelayCalibrationRejected('Delay confidence interval exceeds the motion registration budget')
    return DelayFit(delay, uncertainty, (delay-uncertainty, delay+uncertainty), rms,
        heldout_rms, direction_bias, len(times), len(unique_cycles), float(np.ptp(position)),
        float(np.ptp(depth)))
