"""Encoder/FK circle diagnostics; no robot communication."""
import numpy as np


def circle_coordinates(points, center, normal, start):
    """Signed radial/plane errors and actual tangential distance, not chord length."""
    points, center, normal, start = (np.asarray(v, float) for v in (points, center, normal, start))
    normal = normal / np.linalg.norm(normal)
    initial = start-center
    radius = np.linalg.norm(initial)
    u = initial/radius
    v = np.cross(normal,u)
    delta = points-center
    plane = delta@normal
    uv = np.column_stack((delta@u,delta@v))
    rho = np.linalg.norm(uv,axis=1)
    if np.any(rho < 1e-8):
        raise ValueError('Circle phase undefined at center')
    theta = np.unwrap(np.arctan2(uv[:,1],uv[:,0]))
    # ∫rho d(theta). Using the measured radius avoids overstating speed when
    # the actual path shrinks inside the desired circle. Signed increments also
    # avoid rectifying encoder jitter into a false positive travel distance.
    tangential = np.r_[0., np.cumsum((rho[1:]+rho[:-1])*.5*np.diff(theta))]
    radial = rho-radius
    tangent = (-np.sin(theta)[:,None]*u + np.cos(theta)[:,None]*v)
    return {'plane_m':plane,'radial_m':radial,'edge_m':np.hypot(radial,plane),
            'theta_rad':theta,'tangential_m':tangential,'uv_m':uv,'tangent':tangent,
            'radius_m':radius,'u':u,'v':v}
