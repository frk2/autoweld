import numpy as np
import pytest

from scripts.circle_metrics import circle_coordinates
from scripts.plot_velocity_windows import running_velocity


def test_circle_average_measures_arc_not_chord_and_unwraps_closure():
    t=np.linspace(0,8,801)
    theta=2*t
    radius=.06
    p=radius*np.column_stack((np.cos(theta),np.sin(theta),np.zeros(len(t))))
    c=circle_coordinates(p,[0,0,0],[0,0,1],[radius,0,0])
    avg=running_velocity(t,c['tangential_m'],np.ones(len(t))*.12,.12*t,1.)
    np.testing.assert_allclose(avg['position_mm_s'],120,atol=1e-10)
    np.testing.assert_allclose(c['edge_m'],0,atol=1e-15)
    assert np.linalg.norm(p[100]-p[0]) < c['tangential_m'][100]


def test_shrunken_tilted_circle_separates_errors_and_actual_speed():
    t=np.linspace(0,4,401)
    theta=t
    # Desired radius60mm; actual55mm at1rad/s and3mm plane error.
    center=np.array([.2,.1,.3])
    p=center+np.column_stack((np.full(len(t),.003),.055*np.cos(theta),.055*np.sin(theta)))
    c=circle_coordinates(p,center,[1,0,0],center+[0,.06,0])
    np.testing.assert_allclose(c['radial_m'],-.005,atol=1e-15)
    np.testing.assert_allclose(c['plane_m'],.003,atol=1e-15)
    np.testing.assert_allclose(c['edge_m'],np.hypot(.005,.003),atol=1e-15)
    assert c['tangential_m'][-1]/t[-1] == pytest.approx(.055)
    assert c['theta_rad'][-1]/t[-1]*c['radius_m'] == pytest.approx(.06)
