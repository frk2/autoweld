import numpy as np
import pytest

from autoweld.scan_surface import SurfaceDepthField


def plane_points(step=.001, z=.155, extent=.030):
    x, y = np.meshgrid(np.arange(-extent, extent + step / 2, step),
                       np.arange(-extent, extent + step / 2, step))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def test_dense_measured_plane_gives_bounded_140mm_standoff_and_support():
    points = plane_points()
    original = points.copy()
    field = SurfaceDepthField(points, np.eye(4), last_safe_camera_z_m=.010)
    estimate = field.query(0., 0.)
    assert estimate.supported
    assert estimate.camera_z_m == pytest.approx(.015, abs=1e-5)
    assert estimate.surface_z_m == pytest.approx(.155, abs=1e-5)
    assert estimate.uncertainty_m < .0002
    assert estimate.support_cells >= 3 and estimate.support_points >= 6
    assert estimate.confidence > .999
    assert field(0., 0.) == estimate.camera_z_m
    assert .140 == pytest.approx(.155 - estimate.camera_z_m, abs=1e-5)
    np.testing.assert_array_equal(points, original)
    assert not field.nodes.flags.writeable


def test_no_return_holes_are_travel_depth_only_without_fabricated_surface():
    field = SurfaceDepthField([], np.eye(4), last_safe_camera_z_m=-.020)
    estimate = field.query(.05, -.05)
    assert not estimate.supported
    assert estimate.camera_z_m == -.020
    assert estimate.surface_z_m is None
    assert estimate.uncertainty_m is None
    assert estimate.support_points == 0
    assert 'last-safe' in estimate.reason
    field = SurfaceDepthField(plane_points(extent=.005), np.eye(4), last_safe_camera_z_m=-.020)
    far = field.query(.050, .050)
    assert not far.supported and far.surface_z_m is None
    assert far.camera_z_m == -.020


def test_original_optical_frame_and_chosen_roll_shift_actual_stripe_footprint():
    # Measured points lie20mm right of original optical origin. Chosen +90° roll
    # maps camera +X stripe offset into original optical +Y.
    baseline = np.eye(4)
    baseline[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    baseline[:3, 3] = [.2, .3, .4]
    chosen = baseline[:3, :3] @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    local = plane_points(extent=.004)
    local[:, 1] += .020
    world = local @ baseline[:3, :3].T + baseline[:3, 3]
    field = SurfaceDepthField(world, baseline, fixed_optical_rotation=chosen,
                              laser_offset_camera_m=(.020, 0, .140), support_radius_m=.010)
    estimate = field.query(0, 0)
    assert estimate.supported
    assert estimate.camera_z_m == pytest.approx(.015, abs=1e-5)
    np.testing.assert_allclose(field.footprint_xy, [0, .020], atol=1e-12)
    assert not field.query(0, -.030).supported


def test_native_registration_uses_measured_tcp_and_handeye():
    baseline, handeye = np.eye(4), np.eye(4)
    handeye[:3, 3] = [.010, 0, .005]
    world = plane_points(extent=.010)
    views = []
    for offset in (-.020, .030):
        measured = np.eye(4)
        measured[:3, 3] = [offset, -.010, .025]
        optical = measured @ handeye
        native = (world - optical[:3, 3]) @ optical[:3, :3]
        views.append({'xyz_m': native, 'base_from_tcp': measured})
    field = SurfaceDepthField.from_native_views(views, baseline, handeye)
    estimate = field.query(0, 0)
    assert estimate.supported
    assert estimate.camera_z_m == pytest.approx(.015, abs=1e-5)
    assert field.state()['input_points'] == len(world) * 2


def test_mixed_bin_and_out_of_envelope_returns_are_rejected_not_clamped():
    mixed = np.tile([.0001, .0001, .14], (20, 1))
    mixed[10:, 2] += .020
    outside = np.tile([.020, .020, .300], (10, 1))
    field = SurfaceDepthField(np.r_[mixed, outside], np.eye(4))
    assert field.state()['rejected_mixed_bins'] == 1
    assert field.state()['out_of_bounds_points'] == 10
    assert field.state()['support_cells'] == 0
    estimate = field.query(0, 0)
    assert not estimate.supported and estimate.surface_z_m is None


def test_depth_spread_and_sparse_support_remain_holes():
    points = plane_points(extent=.010)
    points[points[:, 0] > 0, 2] += .040
    field = SurfaceDepthField(points, np.eye(4), max_uncertainty_m=.003)
    estimate = field.query(0, 0)
    assert not estimate.supported
    assert estimate.surface_z_m is None
    assert estimate.uncertainty_m > .003
    assert 'uncertainty' in estimate.reason
    single = SurfaceDepthField([[0., 0., .155]], np.eye(4))
    assert not single.query(0, 0).supported
    assert single.query(0, 0).support_points == 1


def test_compact_kernel_has_no_jump_at_support_edge_and_stays_bounded():
    field = SurfaceDepthField([[0., 0., .180]], np.eye(4), last_safe_camera_z_m=-.040)
    r = field.radius
    # C4 compact support: approaching boundary changes depth only to high order.
    samples = [field(r - h, 0) - field(r + h, 0) for h in (.001, .0005, .00025)]
    assert abs(samples[1]) < abs(samples[0]) / 20
    assert abs(samples[2]) < abs(samples[1]) / 20
    assert field(r, 0) == field.fallback
    for x in np.linspace(-.050, .050, 301):
        assert -.060 <= field(x, 0) <= .060


def test_five_sparse_scout_rows_report_observed_neighbourhoods_and_gaps():
    # Conservative carrier rows; all points remain actual samples. Adjacent25mm
    # rows may support interpolation but no samples beyond footprint support.
    points = []
    for y in np.linspace(-.050, .050, 5):
        for x in np.linspace(-.050, .050, 101):
            for d in np.linspace(-.010, .010, 11):
                points.append([x + d, y - d, .155 + .02 * x])
    field = SurfaceDepthField(points, np.eye(4), max_uncertainty_m=.003)
    assert field.query(0, 0).supported
    assert field.query(.030, .012).supported
    gap = field.query(.100, .100)
    assert not gap.supported and gap.surface_z_m is None


@pytest.mark.parametrize('kwargs', [dict(last_safe_camera_z_m=.061), dict(z_bounds_m=(-.061, .06)),
                                    dict(support_radius_m=0), dict(prior_weight=np.nan),
                                    dict(laser_offset_camera_m=(np.nan, 0, 0)),
                                    dict(fixed_optical_rotation=np.diag([1., -1., -1.]))])
def test_invalid_field_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        SurfaceDepthField([], np.eye(4), **kwargs)
