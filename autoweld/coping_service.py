"""Autonomous coped-joint orchestration on the existing stationary weld owner.

The SDK fits camera-frame geometry. This mixin registers immutable acquisitions
in a frozen seed-camera frame, plans real camera viewpoints, and counts only
fresh measured seam roots as observed angular coverage.
"""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import transform
from autoweld.coping_plan import laser_seam_ray, measured_angle_deg, plan_coping_views, unit


class CopingObservationRejected(ValueError):
    pass


class CopedJointMixin:
    def _joint_init(self):
        self._joint_model = {'status': 'idle', 'quality': {}, 'model': None}
        self._coping_phase = 'idle'
        self._coping_seed_frame = self._coping_seed_mm = self._coping_plane = None
        self._coping_profiles = []
        self._coping_extra_profiles = []
        self._coping_previous = None
        self._coping_scouts = []
        self._coping_scout_index = 0
        self._coping_scout_candidates = {}
        self._coping_scout_baseline_gain_db = None
        self._coping_measured_views = {}
        self._coping_plan = self._coping_curve = self._coping_current = None
        self._coping_indices = {1: 0, -1: 0}
        self._coping_observed = {1: 0., -1: 0.}
        self._coping_boundaries = {}
        self._coping_view_attempt = 0
        self._coping_started = None
        self._coping_bootstrap = {'used': False, 'reason': 'No prior archive selected'}
        self._coping_restored_coverage = None

    @property
    def _is_coping(self):
        return self.settings.joint_kind == 'coped_tube'

    def _joint_start(self, previous):
        self._joint_init()
        # Historical XYZ is opt-in: the physical fixture may have moved even
        # when optical/hand-eye/model identities remain unchanged.
        self._coping_previous = None
        if self.settings.bootstrap_manifest:
            manifest = Path(self.settings.bootstrap_manifest).resolve()
            root = (Path(self.service.runtime)/'weld-paths').resolve()
            if manifest.name != 'path.json' or root not in manifest.parents:
                raise ValueError('Coping bootstrap must be an existing path.json inside this runtime weld-paths archive')
            self._coping_previous = json.loads(manifest.read_text())
        self._coping_started = self.clock()
        if self._is_coping:
            self._coping_phase = 'seed'
            self._joint_model['status'] = 'awaiting_fresh_seed'

    def _joint_measurement(self, measurement):
        if self._is_coping:
            measurement.update(station_basis='coped_angular_stations', station_index=self._distance.get(self._direction, 0),
                angular_theta_deg=None if self._coping_current is None else self._coping_current['angle_deg'],
                coping_phase=self._coping_phase)
            if self._coping_current is not None and self._coping_current.get('seed_recheck'):
                measurement.update(direction=0, distance_mm=0., station_index=0)

    def _joint_previous_point(self, camera_pose):
        if not self._is_coping or self._coping_current is None:
            return None
        expected = np.asarray(self._coping_current['predicted_point_base_m'])
        return (camera_pose[:3, :3].T @ (expected-camera_pose[:3, 3])*1000.).tolist()

    def _joint_check_envelope(self, pose):
        replaying = (self.status in ('replay_preflight', 'replay_wait_feedback')
                     or self._after_settle == 'replay_next' and self.status not in self.TERMINAL)
        if self._is_coping and replaying:
            if not any(np.allclose(pose, candidate, atol=1e-12, rtol=0) for candidate in self._replay):
                raise ValueError('Coping replay target differs from its own complete fresh replay proof')
            return True
        if not self._is_coping or self._coping_phase not in ('orbit', 'returning', 'finishing'):
            return False
        # A 150 mm sensing orbit can be >250 mm from its first camera origin.
        # Admit only an EXACT pose from the preflight artifact (or captured seed
        # return), never a general expansion of the legacy scan envelope.
        known = [self._center['measured_base_from_tcp']]
        if self._coping_plan:
            for station in [self._coping_plan['center'], *self._coping_plan['branches']['positive'],
                            *self._coping_plan['branches']['negative']]:
                known.append(station['base_from_tcp'])
                known.extend(item['base_from_tcp'] for item in station['alternatives'])
        if not any(np.allclose(pose, candidate, atol=1e-12, rtol=0) for candidate in known):
            raise ValueError('Coping motion is outside its exact full-pose preflight artifact')
        return True

    def _coping_axes_base(self, model=None):
        model = model or self._joint_model['model']
        frame = self._coping_seed_frame
        return [dict(origin_m=(frame[:3, :3] @ (np.asarray(axis['origin_mm'])/1000.)+frame[:3, 3]).tolist(),
            direction=(frame[:3, :3] @ unit(axis['direction'])).tolist(), radius_m=axis['radius_mm']/1000.)
            for axis in model['axes']]

    def _joint_move(self, pose):
        if not self._is_coping or self._coping_phase not in ('orbit', 'returning', 'finishing'):
            return False
        previous = getattr(self.service.ctrl, 'motion_collision', None)
        if not callable(previous):
            return False
        from autoweld.coping_clearance import check_tube_clearance
        tubes = self._coping_axes_base()

        def collision(q):
            if previous(q):
                return True
            diagnostic = check_tube_clearance(self.service.planner.model, self.service.planner.data, tubes)
            if diagnostic is not None:
                error = ValueError('Conservative fitted-tube clearance rejected the actual planned coping motion')
                error.diagnostic = diagnostic
                raise error
            return False
        self.service.ctrl.motion_collision = collision
        try:
            self.service.move_pose(pose, automatic=False)
        finally:
            self.service.ctrl.motion_collision = previous
        return True

    def _joint_advance(self):
        if not self._is_coping:
            return False
        if self._coping_phase == 'seed':
            measurement = self._pending_measurement
            self._coping_seed_frame = transform(measurement['base_from_camera'])
            self._coping_seed_mm = np.asarray(self._pending_result['seam_point_mm'], float)
            camera = self.service.camera.state(include_frame=False)
            gain = (measurement['cloud'].get('controls') or camera.get('controls') or {}).get('gain_db')
            if type(gain) in (int, float) and np.isfinite(gain):
                self._coping_scout_baseline_gain_db = float(gain)
            self._coping_plane = np.asarray(camera['intrinsics']['laser_plane'], float).tolist()
            # This first native root locates the assembly. Its shorter range
            # cannot count as a qualified 150 mm surface view or fit sample.
            self._coping_load_prior()
            self._coping_scouts = self._coping_scout_targets()
            self._coping_phase, self._coping_scout_index = 'scout', 0
            self._joint_model['status'] = 'fresh_scout'
            self._move(self._coping_scouts[0]['base_from_tcp'])
            self._coping_current = self._coping_scouts[0]
        elif self._coping_phase == 'orbit':
            self._coping_next_station()
        return True

    def _coping_collect(self, measurement, result, *, fresh):
        native = np.asarray(measurement['cloud']['xyz_mm'], float)
        measured = transform(measurement['base_from_camera'])
        frozen_from_camera = np.linalg.inv(self._coping_seed_frame) @ measured
        points = native @ frozen_from_camera[:3, :3].T+frozen_from_camera[:3, 3]*1000.
        seam = frozen_from_camera[:3, :3] @ np.asarray(result['seam_point_mm'])+frozen_from_camera[:3, 3]*1000.
        profile = dict(xyz_mm=points.tolist(), seam_point_mm=seam.tolist(), profile_id=str(self._pending_native_path),
            provenance=dict(native_path=str(self._pending_native_path), native_sha256=self._native_files[-1]['sha256'],
                fresh_current_run=bool(fresh), source_id=measurement['cloud']['source_id'],
                calibration_id=measurement['cloud']['calibration_id'],
                optical_intrinsics_id=measurement['cloud']['optical_intrinsics_id'],
                acquisition_session_id=measurement['cloud']['acquisition_session_id'],
                base_from_camera=measurement['base_from_camera']))
        self._coping_profiles.append(profile)
        if fresh and self._coping_phase == 'scout':
            display = np.asarray(self._display_profile_cloud(measurement['cloud'])['xyz_mm'], float).reshape(-1, 3)
            world = display/1000. @ measured[:3, :3].T+measured[:3, 3]
            self._coping_extra_profiles.extend(world.round(6).tolist())
            self.revision += 1

    def _joint_scout_surface(self, measurement, result):
        """Retain actual surface returns independently of flat-root visibility.

        A predicted root splits native ordered returns only to initialize the
        cylinder mixture. Neither that hint nor the split is a measured seam.
        All timing, identity, Hold and post-capture checks precede this hook.
        """
        if not self._is_coping or self._coping_phase != 'scout':
            return
        if result.get('status') not in ('found', 'no_seam', 'invalid', 'ambiguous'):
            return
        measured = transform(measurement['base_from_camera'])
        predicted = np.asarray(self._coping_current['predicted_point_base_m'], float)
        if abs(np.linalg.norm(predicted-measured[:3, 3])-.150) > .002:
            return
        hint = measured[:3, :3].T @ (predicted-measured[:3, 3])*1000.
        source = measurement['cloud']
        source_path = 'cloud'
        if self.settings.reflective_surface:
            # The narrow seam ROI can contain a reflection while hiding the
            # useful tube flank. Fit supported native full-profile surfaces;
            # keep the uncertain junction interval out of the surface fit.
            self._display_profile_cloud(source)
            source = source.get('full_profile')
            if not isinstance(source, dict) or not source.get('reflection'):
                return
            source_path = 'full_profile'
            selection = self.service.camera.prepare_coped_surface_profile(source,
                max_profile_gap_mm=self.settings.max_profile_gap_mm, search_hint_mm=hint.tolist())
            if selection.get('status') != 'selected':
                return
        else:
            selection = self.service.camera.select_coped_surface_returns(source)
        native = np.asarray(source.get('xyz_mm'), float)
        if native.ndim != 2 or native.shape[1:] != (3,) or len(native) < 16 or not np.isfinite(native).all():
            return
        raw_native = native
        indices = np.asarray(selection['native_point_indices'], int)
        if (indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(raw_native))
                or len(indices) > 1 and np.any(np.diff(indices) <= 0)):
            raise ValueError('Coping surface selection must reference ordered distinct actual native returns')
        native = raw_native[indices]
        if len(native) < 16:
            return
        frozen = np.linalg.inv(self._coping_seed_frame) @ measured
        if result.get('status') == 'found' and self._coping_scout_index not in self._coping_measured_views:
            root = np.asarray(result.get('seam_point_mm'), float)
            if root.shape == (3,) and np.isfinite(root).all() and abs(np.linalg.norm(root)-150.) <= 2.:
                registered = frozen[:3, :3] @ root+frozen[:3, 3]*1000.
                if np.linalg.norm(registered-self._coping_seed_mm) <= 4.:
                    self._coping_measured_views[self._coping_scout_index] = dict(
                        base_from_camera=measured.tolist(), seam_point_camera_mm=root.tolist(),
                        seam_point_seed_camera_mm=registered.tolist(),
                        native_path=str(self._pending_native_path), native_sha256=self._native_files[-1]['sha256'],
                        physical_scout_index=self._coping_scout_index)
        split = int(np.argmin(np.linalg.norm(native-hint, axis=1)))
        surface_labels = np.asarray(selection.get('surface_labels',
            (np.arange(len(native)) >= split).astype(int)), int)
        if surface_labels.shape != (len(native),) or np.any((surface_labels < 0) | (surface_labels > 1)):
            raise ValueError('Coping surface labels must reference the selected native returns')
        balance = min(int(np.sum(surface_labels == label)) for label in (0, 1))
        if balance < 8:
            return
        old = self._coping_scout_candidates.get(self._coping_scout_index)
        if old is not None and old['provenance']['balanced_surface_point_count'] >= balance:
            return
        points = native @ frozen[:3, :3].T+frozen[:3, 3]*1000.
        provenance = dict(native_path=str(self._pending_native_path), native_sha256=self._native_files[-1]['sha256'],
            fresh_current_run=True, physical_scout_index=self._coping_scout_index,
            source_id=measurement['cloud']['source_id'], calibration_id=measurement['cloud']['calibration_id'],
            optical_intrinsics_id=measurement['cloud']['optical_intrinsics_id'],
            acquisition_session_id=measurement['cloud']['acquisition_session_id'],
            base_from_camera=measurement['base_from_camera'], local_feature_status=result.get('status'),
            split_hint_camera_mm=hint.tolist(), split_hint_is_measured_seam=False,
            membership_basis='predicted membership hint, not measured seam',
            balanced_surface_point_count=balance, native_point_count=len(native))
        provenance.update(raw_native_point_count=len(raw_native), native_point_indices=indices.tolist(),
                          native_cloud_path=source_path)
        if source_path == 'full_profile':
            provenance['surface_profile_selection'] = copy.deepcopy(selection)
        else:
            provenance['surface_return_selection'] = copy.deepcopy(selection['diagnostics'])
        if result.get('status') == 'found':
            provenance['actual_detected_root_camera_mm'] = copy.deepcopy(result.get('seam_point_mm'))
        display = np.asarray(self._display_profile_cloud(measurement['cloud'])['xyz_mm'], float).reshape(-1, 3)
        self._coping_scout_candidates[self._coping_scout_index] = dict(xyz_mm=points.tolist(),
            surface_labels=surface_labels.tolist(),
            profile_id=str(self._pending_native_path), provenance=provenance,
            _display_points_base_m=(display/1000. @ measured[:3, :3].T+measured[:3, 3]).round(6).tolist())

    def _coping_accept_scout_surface(self):
        candidate = self._coping_scout_candidates.pop(self._coping_scout_index, None)
        if candidate is None:
            return False
        display = candidate.pop('_display_points_base_m')
        self._coping_profiles.append(candidate)
        for entry in self._native_files:
            if entry['path'] == candidate['profile_id']:
                entry['disposition'] = 'coping_surface_profile'
                entry['surface_profile_role'] = 'fit geometry only; no measured seam or angular coverage'
        self._coping_extra_profiles.extend(display)
        self.revision += 1
        return True

    def _coping_advance_scout(self):
        self._coping_scout_index += 1
        if self._coping_scout_index < 3:
            self._coping_current = self._coping_scouts[self._coping_scout_index]
            self._move(self._coping_current['base_from_tcp'])
        else:
            self._coping_submit_fit()

    def _coping_load_prior(self):
        prior = self._coping_previous
        if not prior:
            return
        binding = prior.get('binding') or {}
        equivalent = set(getattr(self.service, 'model_equivalent_fingerprints', ())) | {self.service.model_fingerprint}
        equivalence = None
        proof_path = Path(__file__).resolve().parents[1]/'evidence/mount-orientation-20261029/dynamics-preservation.json'
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            if (proof.get('unchanged_robot_dynamics_and_motion_tcp') is True
                    and proof.get('after_model_sha256') == self.service.model_fingerprint
                    and proof.get('before_model_sha256') == binding.get('model_fingerprint')):
                equivalent.add(binding['model_fingerprint'])
                equivalence = dict(path=str(proof_path), sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())
        if (binding.get('model_fingerprint') not in equivalent or binding.get('source_id') != self._identity[0]
                or binding.get('optical_intrinsics_id') != self._identity[1]
                or binding.get('calibration_id') != self._identity[2]
                or not np.array_equal(np.asarray(binding.get('tcp_from_camera')), self._handeye)):
            self._coping_bootstrap = {'used': False, 'reason': 'Prior model/optics/laser/hand-eye identity is not equivalent'}
            return
        index = {entry['path']: entry for entry in prior.get('native_files', [])}
        selected = prior.get('path', [])
        stride = max(1, len(selected)//32)
        loaded = 0
        before = len(self._coping_profiles)
        for record in selected[::stride][:32]:
            path = record.get('native_path')
            entry = index.get(path) or {}
            try:
                raw = Path(path).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry.get('sha256') or record.get('native_sha256') != entry['sha256']:
                    raise ValueError('Prior native bytes fail their retained hash')
                payload = json.loads(raw)
                if payload['result'].get('status') != 'found':
                    continue
                cloud = payload['measurement']['cloud']
                if any(cloud.get(key) != binding.get(key) for key in
                       ('source_id', 'optical_intrinsics_id', 'calibration_id')):
                    raise ValueError('Prior native acquisition differs from its camera calibration binding')
                if not np.allclose(transform(payload['measurement']['base_from_tcp']) @ self._handeye,
                                   transform(payload['measurement']['base_from_camera']), atol=1e-10, rtol=0):
                    raise ValueError('Prior measured camera pose is inconsistent with unchanged hand-eye')
                original_path, original_file = self._pending_native_path, self._native_files[-1]
                self._pending_native_path = Path(path)
                self._native_files.append(entry)
                try:
                    self._coping_collect(payload['measurement'], payload['result'], fresh=False)
                finally:
                    self._native_files.pop()
                    self._pending_native_path = original_path
                    assert self._native_files[-1] is original_file
                loaded += 1
            except (OSError, ValueError, TypeError, KeyError) as error:
                del self._coping_profiles[before:]
                self._coping_bootstrap = {'used': False, 'reason': str(error)}
                return
        self._coping_bootstrap = dict(used=bool(loaded), profile_count=loaded, role='fit prior only; never current observed coverage',
                                      prior_model_fingerprint=binding['model_fingerprint'], model_equivalence_proof=equivalence)

    def _coping_scout_targets(self):
        camera = self._coping_seed_frame
        point = camera[:3, :3] @ (self._coping_seed_mm/1000.)+camera[:3, 3]
        tangent = unit(camera[:3, :3] @ np.asarray(self._coping_plane)[:3])
        ray = laser_seam_ray(self._coping_plane, 150.)/1000.
        targets = []
        for distance, angle in ((0., 0.), (3., 5.), (-3., -5.), (5., 8.), (-5., -8.),
                                (7., 8.), (-7., -8.), (9., 10.), (-9., -10.)):
            target_point = point+tangent*distance/1000.
            view = camera.copy()
            view[:3, :3] = Rotation.from_rotvec(tangent*np.deg2rad(angle)).as_matrix() @ camera[:3, :3]
            view[:3, 3] = target_point-view[:3, :3] @ ray
            pose = view @ np.linalg.inv(self._handeye)
            targets.append(dict(base_from_tcp=pose.tolist(), predicted_point_base_m=target_point.tolist(),
                                angle_deg=0., scout_offset_mm=distance, scout_view_tilt_deg=angle))
        return targets

    def _joint_process_found(self, measurement, result):
        if not self._is_coping:
            return False
        if self._coping_phase == 'seed':
            camera = transform(measurement['base_from_camera'])
            seam = np.asarray(result['seam_point_mm'], float)
            if seam.shape != (3,) or not np.isfinite(seam).all() or seam[2] <= 0:
                raise ValueError('Initial coping scout requires a finite positive-depth measured root')
            torch = transform(result['camera_from_torch'])
            torch[:3, 3] /= 1000.
            point = camera[:3, :3] @ (seam/1000.)+camera[:3, 3]
            record = dict(direction=0, distance_mm=0., station_index=0,
                station_basis='coping_seed_scout', provisional_seed_scout=True,
                seam_point_base_m=point.tolist(), measured_base_from_tcp=measurement['base_from_tcp'],
                scan_lines_base_m=(np.asarray(result['flank_lines_mm'])/1000. @ camera[:3, :3].T+camera[:3, 3]).tolist(),
                torch_pose_base_m=(camera @ torch).tolist(), acquisition_key=measurement['acquisition_key'],
                quality=result.get('quality'), detector=result, native_path=str(self._pending_native_path),
                native_sha256=self._native_files[-1]['sha256'])
            self._native_files[-1]['disposition'] = 'coping_scout'
            self.records.append(record)
            self._center = record
            self._scan_direction = unit(camera[:3, :3] @ np.asarray(result['scan_direction_camera']))
            self._reference_direction = self._scan_direction.copy()
            self._retain_profile(record, measurement)
            self.revision += 1
            self._joint_advance()
            return True
        if self._coping_phase not in ('scout', 'orbit'):
            return False
        if self._coping_phase == 'scout':
            self._native_files[-1]['disposition'] = 'coping_surface_candidate'
            # A local root can be found on a small or glare-biased subset.
            # Finish the bounded native surface bracket before selecting one
            # balanced profile; successful valley detection is not fit quality.
            self._begin_optical_recovery('Comparing native scout surfaces across the fixed baseline exposure/gain bracket')
            return True
        try:
            record = self._coping_record(measurement, result)
        except CopingObservationRejected as error:
            self._native_files[-1]['disposition'] = 'coping_unassociated_root'
            # A false/glare corner does not establish a coverage boundary.
            # Reuse the bounded same-Hold bracket before giving up this view.
            self._begin_optical_recovery(str(error))
            return True
        self._native_files[-1]['disposition'] = 'path_point'
        if self._coping_current.get('seed_recheck'):
            if self._direction == 1:
                old = self.records[0]
                self._coping_extra_profiles.extend(self._profiles.pop(old['native_path'], []))
                for entry in self._native_files:
                    if entry['path'] == old['native_path']:
                        entry['disposition'] = 'coping_scout'
                record.update(direction=0, distance_mm=0., station_index=0)
                self.records[0] = record
            else:
                self._native_files[-1]['disposition'] = 'coping_scout'
                self._coping_extra_profiles.extend((np.asarray(self._display_profile_cloud(measurement['cloud'])['xyz_mm']).reshape(-1, 3)/1000.
                    @ np.asarray(measurement['base_from_camera'])[:3, :3].T
                    +np.asarray(measurement['base_from_camera'])[:3, 3]).round(6).tolist())
                self._coping_next_station()
                return True
        else:
            self.records.append(record)
        self._retain_profile(record, measurement)
        self.revision += 1
        angle = record['angular_theta_deg']
        if record['direction']:
            self._coping_observed[self._direction] = max(self._coping_observed[self._direction], self._direction*angle)
        self._coping_next_station()
        return True

    def _coping_submit_fit(self):
        if sum(profile['provenance']['fresh_current_run'] for profile in self._coping_profiles) < 3:
            if self._coping_scout_index >= len(self._coping_scouts):
                raise ValueError('Fewer than three distinct fresh 150 mm surface views support the coping model')
            self._coping_phase = 'scout'
            self._coping_current = self._coping_scouts[self._coping_scout_index]
            self._move(self._coping_current['base_from_tcp'])
            return
        self._coping_phase = 'fitting'
        self._joint_model['status'] = 'fitting'
        self._submit('coping_fit', self.service.camera.fit_coped_joint, copy.deepcopy(self._coping_profiles),
            coordinate_frame='camera', radii_mm=((self.settings.branch_diameter_mm/2, self.settings.main_diameter_mm/2)
                if self.settings.coping_axis == 0 else (self.settings.main_diameter_mm/2, self.settings.branch_diameter_mm/2)),
            seam_seed_mm=self._coping_seed_mm.tolist(), joint_angle_deg=self.settings.joint_angle_deg,
            axis_hints=getattr(self.service, 'coping_axis_hints_seed_camera', None))
        self.status, self.message = 'coping_fitting', 'Fitting measured coped-tube profiles in the frozen seed-camera frame'

    def _coping_validate_fresh(self, fitted):
        classification = fitted.get('classification') or {}
        labels = np.asarray(classification.get('surface_labels', []), int)
        groups = np.asarray(classification.get('profile_indices', []), int)
        inlier = np.asarray(classification.get('inlier_mask', []), bool)
        residuals = np.asarray(classification.get('residuals_mm', []), float)
        if (not len(labels) or groups.shape != labels.shape or inlier.shape != labels.shape
                or residuals.shape != (len(labels), 2) or not np.isfinite(residuals).all()
                or np.any((labels < 0) | (labels > 1))):
            return dict(qualified=False, reason='Coping fit lacks per-return classification needed to verify CURRENT-run support')
        fresh = [index for index, profile in enumerate(self._coping_profiles) if profile['provenance']['fresh_current_run']]
        chosen = abs(residuals[np.arange(len(labels)), labels])
        reports = []
        blockers = []
        for index in fresh:
            mask = groups == index
            fraction = float(np.mean(inlier[mask])) if mask.any() else 0.
            rms = float(np.sqrt(np.mean(chosen[mask]**2))) if mask.any() else None
            counts = [int(np.sum(mask & inlier & (labels == surface))) for surface in (0, 1)]
            reports.append(dict(profile_index=index, inlier_fraction=fraction, all_point_rms_mm=rms,
                                per_surface_inlier_count=counts))
            if fraction < .7 or rms is None or rms > .5 or min(counts) < 10:
                blockers.append(f'Fresh profile {index} lacks independently supported two-surface geometry')
        mask = np.isin(groups, fresh)
        counts = [int(np.sum(mask & inlier & (labels == surface))) for surface in (0, 1)]
        if len(fresh) < 3 or min(counts) < 50:
            blockers.append('At least three CURRENT profiles and 50 current inliers per cylinder are required')
        return dict(qualified=not blockers, reason='; '.join(blockers) if blockers else
                    'CURRENT-run profiles independently support both fitted surfaces', profiles=reports,
                    per_surface_inlier_count=counts, prior_cannot_outvote_current=True)

    def _joint_future(self, kind, value):
        if kind == 'coping_fit':
            self._joint_model = copy.deepcopy(value)
            self._joint_model['bootstrap'] = copy.deepcopy(self._coping_bootstrap)
            self._joint_model['branch_identity'] = dict(coping_axis=self.settings.coping_axis,
                selected_side='left' if self.settings.coping_axis == 0 else 'right',
                basis='selected side of ordered seed laser profile',
                source='explicit caller selection; geometry alone retains ambiguous physical assembly roles')
            if value.get('status') == 'qualified':
                current = self._coping_validate_fresh(value)
                self._joint_model['current_validation'] = current
                if not current['qualified']:
                    self._joint_model['status'] = 'unobservable'
                    self._joint_model.setdefault('quality', {})['reason'] = current['reason']
                    value = self._joint_model
            if value.get('status') != 'qualified':
                if self._coping_scout_index < len(self._coping_scouts):
                    self._coping_phase = 'scout'
                    self._coping_current = self._coping_scouts[self._coping_scout_index]
                    self._move(self._coping_current['base_from_tcp'])
                    return True
                raise ValueError('Measured coped-tube model is not qualified: '+str(value.get('quality')))
            self._coping_pending_model = copy.deepcopy(value['model'])
            self._coping_admission_deadline = self.clock()+5.
            self.status = 'coping_fit_admission'
            self._joint_admission_tick()
            return True
        if kind == 'coping_plan':
            self._coping_pending_plan = value
            self._coping_admission_deadline = self.clock()+5.
            self.status = 'coping_plan_admission'
            self._joint_admission_tick()
            return True
        return False

    def _joint_admission_tick(self):
        camera, feedback, _ = self._sample()
        self._camera_identity(camera)
        if not camera.get('enabled') or camera.get('error'):
            raise ValueError('Camera disconnected or failed during coping model/preflight work; no orbit move submitted')
        if self.clock() > self._coping_admission_deadline:
            raise TimeoutError('Coping plan admission did not obtain fresh stationary feedback within five seconds')
        if (not feedback.get('fresh') or not feedback.get('stationary')
                or not self.service.ctrl.ready_for_waypoint()
                or getattr(self.service.ctrl, 'mode', 'Hold') != 'Hold'):
            self.message = 'Waiting at the same Hold for fresh stationary coping-plan admission'
            return
        if self.status == 'coping_fit_admission':
            self._coping_phase = 'preflight'
            source = copy.copy(self.service.ctrl.planner_for_controller())
            source.limits = copy.copy(source.limits)
            seed = self.service.ctrl.positions().copy()
            bounds = self.service.ctrl.effective_joint_bounds().copy()
            self._coping_proof_binding = dict(start_q=seed.tolist(), bounds=bounds.tolist(), limits=asdict(source.limits),
                camera_identity=list(self._identity), model_fingerprint=self.service.model_fingerprint)
            self._submit('coping_plan', self._coping_build_plan, self._coping_pending_model, seed, source, bounds)
            self.status, self.message = 'coping_preflight', 'Preflighting full camera poses for both coping arc directions'
        else:
            binding = self._coping_proof_binding
            if (self.service.model_fingerprint != binding['model_fingerprint']
                    or list(self._identity) != binding['camera_identity']
                    or not np.array_equal(self.service.ctrl.effective_joint_bounds(), binding['bounds'])
                    or asdict(self.service.ctrl.planner_for_controller().limits) != binding['limits']
                    or np.max(abs(self.service.ctrl.positions()[:6]-np.asarray(binding['start_q'])[:6])) > .01):
                raise ValueError('Camera/model, limits, bounds or measured arm pose changed during coping preflight')
            value = self._coping_pending_plan
            self._coping_plan, self._coping_curve = value['plan'], value['curve']
            self._coping_plan['admission_binding'] = binding
            self.revision += 1
            self._coping_boundaries.update(copy.deepcopy(self._coping_plan['boundaries']))
            self._coping_phase, self._direction = 'orbit', 1
            self._coping_current = self._coping_plan['center']
            # The initial root remains a scout until a fresh acquisition at the
            # preflighted 150 mm view replaces it with an admitted orbit point.
            self._move(self._coping_current['base_from_tcp'])
            self._coping_current = {**self._coping_current, 'seed_recheck': True}

    def _coping_build_plan(self, model, start_q, planner, bounds):
        radius = self.settings.branch_diameter_mm/2.
        step = float(np.clip(np.rad2deg(1./radius), 2., 5.))
        intervals_each_side = int(np.ceil(self.settings.target_arc_deg/2./step))
        step = self.settings.target_arc_deg/(2*intervals_each_side)*(1+1e-12)
        curve = self.service.camera.sample_coping_curve(model, self._coping_seed_mm.tolist(),
            coping_axis=self.settings.coping_axis, span_deg=self.settings.target_arc_deg,
            step_deg=step, direction_hint=None, root_sign=getattr(self.service, 'coping_root_sign', None))
        if curve.get('status') not in ('complete', 'partial') or not curve.get('predicted_only'):
            raise ValueError('Coping curve branch is not observable/qualified: '+str(curve.get('quality')))
        reference = None
        for measured in self._coping_measured_views.values():
            association = self.service.camera.project_coping_seam(model,
                [measured['seam_point_seed_camera_mm']], self._coping_seed_mm.tolist(),
                coping_axis=self.settings.coping_axis, max_distance_mm=1., root_sign=curve['root_sign'])['associations'][0]
            if association['associated'] and not association.get('singularity'):
                reference = copy.deepcopy(measured)
                reference['association'] = association
                break
        proof = plan_coping_views(planner, start_q, bounds,
            self._coping_seed_frame, self._handeye, self._coping_plane, curve,
            cancelled=self._cancel_event.is_set, fitted_tubes=self._coping_axes_base(model),
            measured_camera_reference=None if reference is None else reference['base_from_camera'])
        proof['measured_view_reference'] = reference
        return dict(plan=proof, curve=curve)

    def _coping_record(self, measurement, result):
        actual = transform(measurement['base_from_camera'])
        point = actual[:3, :3] @ (np.asarray(result['seam_point_mm'])/1000.)+actual[:3, 3]
        expected = np.asarray(self._coping_current['predicted_point_base_m'])
        if np.linalg.norm(point-expected) > .004:
            raise CopingObservationRejected('Measured coping root is farther than 4 mm from its model association; refusing surface switch')
        if abs(np.linalg.norm(point-actual[:3, 3])-.150) > .002:
            raise ValueError('Measured coping seam is outside 150 ± 2 mm sensing clearance')
        record = dict(direction=self._direction, distance_mm=self._distance[self._direction],
            station_basis='coped_angular_stations', station_index=self._distance[self._direction],
            seam_point_base_m=point.tolist(), measured_base_from_tcp=measurement['base_from_tcp'],
            scan_lines_base_m=(np.asarray(result['flank_lines_mm'])/1000. @ actual[:3, :3].T+actual[:3, 3]).tolist(),
            acquisition_key=measurement['acquisition_key'], quality=result.get('quality'), detector=result,
            native_path=str(self._pending_native_path), native_sha256=self._native_files[-1]['sha256'])
        return self._coping_orient_record(record)

    def _coping_orient_record(self, record):
        record = copy.deepcopy(record)
        inverse = np.linalg.inv(self._coping_seed_frame)
        point = inverse[:3, :3] @ (np.asarray(record['seam_point_base_m'])-self._coping_seed_frame[:3, 3])*1000.
        model = self._joint_model['model']
        axis_index = self.settings.coping_axis
        axis = model['axes'][axis_index]
        origin, direction = np.asarray(axis['origin_mm']), unit(axis['direction'])
        closest = origin+direction*np.dot(point-origin, direction)
        inward = unit(closest-point)
        radial_error = abs(np.linalg.norm(point-closest)-axis['radius_mm'])
        other = model['axes'][1-axis_index]
        od = unit(other['direction'])
        delta = point-np.asarray(other['origin_mm'])
        other_error = abs(np.linalg.norm(delta-od*np.dot(delta, od))-other['radius_mm'])
        if max(radial_error, other_error) > 1.:
            raise CopingObservationRejected('Measured coping seam is more than 1 mm from the qualified cylinder intersection')
        beta_cosine = float(direction @ od)
        beta_sine = float(np.sqrt(max(0., 1.-beta_cosine**2)))
        e = unit(od-beta_cosine*direction)
        local = point-origin
        finite_root_coordinate = beta_sine*np.dot(local, direction)-beta_cosine*np.dot(local, e)
        root_sign = self._coping_curve['root_sign']
        uncertainty_band = max(.2, min(1., 3.*float(self._joint_model.get('quality', {}).get('heldout_robust_rms_mm') or .2)))
        if finite_root_coordinate*root_sign <= uncertainty_band:
            raise CopingObservationRejected('Selected coping root sign is indistinguishable or reversed within the fitted uncertainty band')
        association = self.service.camera.project_coping_seam(model, [point.tolist()], self._coping_seed_mm.tolist(),
            coping_axis=axis_index, max_distance_mm=1., root_sign=self._coping_curve['root_sign'])['associations'][0]
        if not association['associated'] or association.get('singularity'):
            raise CopingObservationRejected('Actual coping root cannot be associated with the selected observable finite intersection branch')
        angle = measured_angle_deg(point, self._coping_seed_mm, axis)
        previous = [row for row in self.records if row['direction'] == self._direction]
        if record['direction'] and self._coping_current is not None and not self._coping_current.get('seed_recheck'):
            before = previous[-1]['angular_theta_deg'] if previous else 0.
            advance = self._direction*(angle-before)
            requested = abs(self._coping_current['angle_deg']-(previous[-1].get('requested_angular_theta_deg', 0.) if previous else 0.))
            if advance <= 1e-6 or advance > max(requested*1.5+1., 3.):
                raise CopingObservationRejected('Actual coping angles contain a duplicate, reversal or unobserved angular gap')
        hint = (None if self._coping_current is None or 'tangent_base' not in self._coping_current else
                self._coping_seed_frame[:3, :3].T @ np.asarray(self._coping_current['tangent_base']))
        torch = self.service.camera.coping_reference_pose(model, point.tolist(), coping_axis=axis_index,
                                                         standoff_mm=150., tangent_hint=hint)
        torch = transform(torch)
        torch[:3, 3] /= 1000.
        record.update(torch_pose_base_m=(self._coping_seed_frame @ torch).tolist(),
            torch_orientation_policy='coped_centerline',
            angular_theta_deg=angle,
            requested_angular_theta_deg=0. if self._coping_current is None else self._coping_current['angle_deg'],
            coping_geometry=dict(model_seed_camera=copy.deepcopy(model),
                base_from_seed_camera=self._coping_seed_frame.tolist(), coping_axis=axis_index,
                measured_point_seed_camera_mm=point.tolist(), centerline_closest_point_seed_camera_mm=closest.tolist(),
                inward_direction_seed_camera=inward.tolist(), tangent_hint_seed_camera=None if hint is None else hint.tolist(),
                radial_residual_mm=[float(radial_error), float(other_error)], association=association,
                root_sign=root_sign, finite_root_signed_coordinate_mm=float(finite_root_coordinate),
                finite_root_ambiguity_band_mm=uncertainty_band))
        return record

    def _coping_next_station(self):
        name = 'positive' if self._direction == 1 else 'negative'
        branch = self._coping_plan['branches'][name]
        index = self._coping_indices[self._direction]
        if index >= len(branch):
            self._coping_finish_direction('preflight_limit' if name in self._coping_plan['boundaries'] else 'target_arc_reached')
            return
        self._coping_current = branch[index]
        self._coping_indices[self._direction] += 1
        self._distance[self._direction] = self._coping_indices[self._direction]
        self._coping_view_attempt = 0
        self._move(self._coping_current['base_from_tcp'])
        self.message = f'Observing coping angle {self._coping_current["angle_deg"]:+.1f}° with a full camera pose'

    def _joint_no_seam(self, measurement):
        if not self._is_coping:
            return False
        if self._coping_phase == 'scout':
            self._begin_optical_recovery('Fresh scout surface did not resolve an actual coping root')
            return True
        if self._coping_phase != 'orbit':
            return False
        self._native_files[-1]['disposition'] = 'coping_view_no_seam'
        self._joint_optical_limit('good_profile_without_observable_coping_root')
        return True

    def _joint_surface_bracket(self, camera, limits):
        """Reset physical scouts to the locator's measured gain, without drift.

        Every candidate is acquired at the same measured Hold through the
        existing owner/configuration/fresh-frame gates. At most five configs
        remain available; high gain from a previous view cannot become the next
        view's baseline or remove its lower-gain alternatives.
        """
        if not self._is_coping or self._coping_phase not in ('scout', 'orbit'):
            return None
        bounds, baseline = self._gain_bounds(camera), self._coping_scout_baseline_gain_db
        if bounds is None or baseline is None:
            return None
        if not bounds[0] <= baseline <= min(bounds[1], 24.):
            raise ValueError('Measured scout baseline gain exceeds actual sensor/recovery bounds')
        if self.settings.reflective_surface:
            return self._reflective_surface_bracket(camera, limits, baseline_gain=baseline)
        high = limits['max_us']
        candidates = [dict(kind='gain', exposure_us=high, gain_db=baseline)]
        candidates.extend(dict(kind='exposure', exposure_us=max(limits['min_us'], high*factor),
                               gain_db=baseline) for factor in (.5, .25))
        gains = []
        for increment in (6., 12.):
            gain = min(baseline+increment, bounds[1], 24.)
            if gain > baseline and gain not in gains:
                gains.append(gain)
                candidates.append(dict(kind='gain', exposure_us=high, gain_db=gain))
        return candidates

    def _joint_optical_limit(self, reason):
        if not self._is_coping:
            return False
        if self._coping_phase == 'scout':
            if self._coping_accept_scout_surface():
                self._coping_advance_scout()
            else:
                self._coping_scout_failed(reason)
            return True
        if self._coping_phase != 'orbit':
            return False
        alternatives = self._coping_current.get('alternatives', [])
        if self._coping_view_attempt < min(2, len(alternatives)):
            alternate = alternatives[self._coping_view_attempt]
            self._coping_view_attempt += 1
            self._move(alternate['base_from_tcp'])
            self.message = 'Trying a preflighted bounded tilted view of the same predicted coping root'
        else:
            self._coping_finish_direction('optical_visibility_unresolved: '+reason)
        return True

    def _coping_scout_failed(self, reason):
        self._native_files[-1]['disposition'] = 'coping_scout_no_root'
        self._joint_model.setdefault('failed_scouts', []).append(dict(index=self._coping_scout_index, reason=reason))
        self._coping_scout_index += 1
        self._coping_submit_fit()

    def _coping_finish_direction(self, reason):
        name = 'positive' if self._direction == 1 else 'negative'
        self._coping_boundaries[name] = dict(reason=reason,
            angle_deg=0. if self._coping_current is None else self._coping_current['angle_deg'],
            observed_bound_deg=self._coping_observed[self._direction]*self._direction,
            kind='coverage_boundary_not_a_physical_seam_endpoint')
        self.endpoints[name] = dict(confirmed=False, reason=reason, angular_bound_deg=self._coping_observed[self._direction]*self._direction)
        if self._direction == 1:
            visited = self._coping_plan['branches']['positive'][:self._coping_indices[1]]
            self._coping_return_queue = [row['base_from_tcp'] for row in reversed(visited)]
            self._coping_return_queue.append(self._coping_plan['center']['base_from_tcp'])
            self._coping_phase = 'returning'
            self._joint_return_step()
        else:
            self._coping_phase = 'finishing'
            self._finish()

    def _joint_return_step(self):
        if self._coping_return_queue:
            self._move(self._coping_return_queue.pop(0), 'coping_return_step')
            self.message = 'Returning through the validated coping viewpoints before the opposite branch'
            return
        self._direction, self._coping_phase = -1, 'orbit'
        self._coping_current = {**self._coping_plan['center'], 'seed_recheck': True}
        self._after_settle, self.status = 'capture', 'capturing'

    def _joint_begin_rest_return(self):
        if not self._is_coping or not callable(getattr(self.service.ctrl, 'return_rest', None)):
            return False
        self._set_phase('coping_returning_rest', 'Coping acquisition saved; returning to verified supported rest')
        previous = getattr(self.service.ctrl, 'motion_collision', None)
        if callable(previous):
            from autoweld.coping_clearance import check_tube_clearance
            tubes = self._coping_axes_base()

            def collision(q):
                if previous(q):
                    return True
                diagnostic = check_tube_clearance(self.service.planner.model, self.service.planner.data, tubes)
                if diagnostic is not None:
                    error = ValueError('Conservative fitted-tube clearance blocks the return to rest')
                    error.diagnostic = diagnostic
                    raise error
                return False
            self.service.ctrl.motion_collision = collision
        try:
            self.service.ctrl.return_rest(automatic=False)
        finally:
            self.service.ctrl.motion_collision = previous
        return True

    def _joint_rest_return_tick(self):
        feedback = self.service.scan_feedback()
        if (feedback.get('fresh') and feedback.get('stationary') and self.service.ctrl.at_rest()
                and self.service.ctrl.rest_verified and self.service.ctrl.ready_for_waypoint()):
            self.status = self._result['status']
            self.message = 'Coping acquisition saved; captured rest verified, motors remain in powered hold'

    def _joint_result(self, result, *, partial):
        result['joint_kind'] = self.settings.joint_kind
        if not self._is_coping:
            result['plate_thickness_mm'] = self.settings.plate_thickness_mm
            return result
        coverage = self._coping_coverage()
        full = not partial and coverage['coverage_complete']
        result.update(status='complete' if full else 'incomplete', coverage_complete=full,
            minimum_satisfied=full, minimum_basis='measured_coped_angular_span',
            replay_scope='full_observed_path' if full else 'measured_section_only',
            joint_model=copy.deepcopy(self._joint_model), coping_coverage=coverage,
            coping_seed_seam_mm=None if self._coping_seed_mm is None else self._coping_seed_mm.tolist(),
            coping_seed_base_from_camera=None if self._coping_seed_frame is None else self._coping_seed_frame.tolist(),
            coping_view_reference=None if self._coping_plan is None else self._coping_plan.get('measured_view_reference'),
            coping_curve=copy.deepcopy(self._coping_curve), coping_profiles=copy.deepcopy(self._coping_profiles))
        if not partial:
            result['finding_status'] = result['status']
        return result

    def _coping_coverage(self):
        if self._coping_restored_coverage is not None:
            return copy.deepcopy(self._coping_restored_coverage)
        observed = self._coping_observed[1]+self._coping_observed[-1]
        return dict(requested_arc_deg=self.settings.target_arc_deg,
            planned_arc_deg=0. if self._coping_plan is None else self._coping_plan['planned_arc_deg'],
            observed_arc_deg=observed, positive_observed_deg=self._coping_observed[1],
            negative_observed_deg=self._coping_observed[-1],
            coverage_complete=all(self._coping_observed[d]+1e-8 >= self.settings.target_arc_deg/2 for d in (1, -1)),
            measured_station_count=sum(not record.get('provisional_seed_scout') for record in self.records),
            provisional_seed_scout_count=sum(bool(record.get('provisional_seed_scout')) for record in self.records),
            boundaries=copy.deepcopy(self._coping_boundaries),
            preflight=None if self._coping_plan is None else {key: self._coping_plan.get(key) for key in
                ('status', 'positive_bound_deg', 'negative_bound_deg', 'sensing_view_family',
                 'base_view_tilt_deg', 'normal_sign')},
            angular_basis='actual measured seam radial angle around the fitted coping-cylinder axis',
            prediction_role='fit and planning guidance only; not observed geometry')

    def _joint_state(self):
        # Per-return assignments/residuals belong to the retained fit report,
        # not 5 Hz telemetry competing with fresh camera/SDK acquisitions.
        summary = {key: self._joint_model[key] for key in ('status', 'model', 'quality', 'bootstrap', 'model_comparison',
            'failed_scouts', 'branch_identity') if key in self._joint_model}
        return dict(joint_kind=self.settings.joint_kind, joint_model=copy.deepcopy(summary),
                    coping_coverage=self._coping_coverage() if self._is_coping else None)

    def _joint_restore(self, result):
        if not self._is_coping or not result:
            return
        self._joint_model = result.get('joint_model', self._joint_model)
        frame = result.get('coping_seed_base_from_camera')
        self._coping_seed_frame = None if frame is None else transform(frame)
        self._coping_curve = result.get('coping_curve')
        self._coping_profiles = result.get('coping_profiles', [])
        coverage = result.get('coping_coverage') or {}
        self._coping_restored_coverage = copy.deepcopy(coverage)
        self._coping_observed = {1: coverage.get('positive_observed_deg', 0.), -1: coverage.get('negative_observed_deg', 0.)}
        self._coping_boundaries = coverage.get('boundaries', {})
        if self._coping_profiles and self._coping_seed_frame is not None:
            seed = result.get('coping_seed_seam_mm', self._coping_profiles[0].get('seam_point_mm'))
            self._coping_seed_mm = None if seed is None else np.asarray(seed)
            for entry in result.get('native_files', []):
                if entry.get('disposition') not in ('coping_scout', 'coping_surface_profile'):
                    continue
                raw = Path(entry['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError('Restored coping scout profile SHA256 mismatch')
                measurement = json.loads(raw)['measurement']
                points = np.asarray(self._display_profile_cloud(measurement['cloud'])['xyz_mm']).reshape(-1, 3)/1000.
                camera = transform(measurement['base_from_camera'])
                self._coping_extra_profiles.extend((points @ camera[:3, :3].T+camera[:3, 3]).round(6).tolist())
