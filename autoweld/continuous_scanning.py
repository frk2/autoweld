"""One-request continuous area scan orchestration through service callbacks.

The running service owns all devices, motion validation, measured settling,
clock/delay experiments, registration and persistence. This state machine only
schedules bounded work and consumes results; it never opens a device or model.
Callbacks must check context['cancel_event'] before every service command and
must retain the normal physical and quality gates in their respective stages.
"""
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass

from autoweld.motion_registration import RegistrationRejected


@dataclass(frozen=True)
class ContinuousScanSettings:
    width_mm: float = 100.
    height_mm: float = 100.
    spacing_mm: float = 1.
    standoff_mm: float = 140.
    optical_roll_deg: float = -45.
    feed_mm_s: float = 30.
    max_duration_s: float = 900.
    start_pose: str = 'Tube Scanning'
    empirical_timing_diagnostic: bool = False
    scan_profile: str = 'area'
    scan_method: str = 'fast'
    sweep_length_mm: float = 80.

    @classmethod
    def from_request(cls, request):
        profile = request.get('scan_profile','area')
        if profile in ('downward_line','line_down','singleline'):
            profile = 'vertical_down'
        method = request.get('scan_method','fast')
        if profile not in ('area','vertical_down') or method not in ('fast','precision') or profile == 'area' and method == 'precision':
            raise ValueError('Choose fast area scan or fast/precision vertical_down')
        values = dict(scan_profile=profile,scan_method=method,sweep_length_mm=request.get('sweep_length_mm',80.),width_mm=request.get('horizontal_mm', 100.), height_mm=request.get('vertical_mm', 100.),
            optical_roll_deg=request.get('optical_roll_deg',0. if profile == 'vertical_down' else -45.), feed_mm_s=request.get('feed_mm_s',10. if profile == 'vertical_down' else 30.),
            max_duration_s=request.get('max_duration_s', 900.), start_pose=request.get('start_pose', 'Tube Scanning'),
            empirical_timing_diagnostic=request.get('empirical_timing_diagnostic',False))
        for key, value in values.items():
            if key not in ('start_pose','empirical_timing_diagnostic','scan_profile','scan_method') and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'{key} must be finite')
        if not 1 <= values['sweep_length_mm'] <= 100 or values['sweep_length_mm'] != round(values['sweep_length_mm']):
            raise ValueError('Vertical sweep length must be whole millimetres in[1,100]')
        if type(values['empirical_timing_diagnostic']) is not bool:
            raise ValueError('Empirical timing diagnostic must be an explicit boolean')
        if not 1 <= values['width_mm'] <= 100 or not 1 <= values['height_mm'] <= 100:
            raise ValueError('Continuous area extents must be between 1 and 100 mm')
        if not -180 <= values['optical_roll_deg'] <= 180:
            raise ValueError('Fixed optical roll must be between -180 and 180 degrees')
        if not 0 < values['feed_mm_s'] <= 30:
            raise ValueError('Continuous scan feed must be in (0,30] mm/s')
        if not 0 < values['max_duration_s'] <= 900:
            raise ValueError('Continuous scan maximum duration must be in (0,900] seconds')
        if values['start_pose'] not in ('Tube Scanning', 'current'):
            raise ValueError('Scan starting pose must be Tube Scanning or current')
        return cls(**values)


class ContinuousScanCoordinator:
    """Nonblocking tick state machine with one stage and one capture in flight.

    Stage callback(context) -> dict or Future[dict]. Results merge into context
    data under their phase name; optional public fields in results update state.
    prepare: existing camera singleton + saved-pose route.
    baseline: good native stripe, 140mm alignment, fixed roll, measured quiet Hold.
    clock_warm: verified native global-shutter exposure clock qualification.
    delay_calibration: bidirectional held-out measured-output delay experiment.
    scout: bounded physical surface scout using the same optical XY square.
    plan: full existing guards; result validated=True, duration_s=<actual path>.
    move_entry: normal service move, endpoint convergence and single measured Hold.
    start_motion: service starts the validated TimedJointPath without new limits.
    finalize: flush, finish and export accepted moving acquisitions.
    return_rest: normal service route, measured endpoint/quiet verification.

    capture(context) registers/archives the latest unique exposure while moving;
    return qualified=True only for a NEW exposure with all timing/feedback gates.
    motion_state(context) is a fast snapshot with active/complete/error; complete
    must mean measured endpoint completion, never just elapsed profile duration.
    release(context) must release scan ownership and retain persisted partials.
    """
    STAGES = ('prepare', 'baseline', 'clock_warm', 'delay_calibration', 'scout', 'plan',
              'move_entry', 'start_motion', 'finalize', 'return_rest')
    DEFAULT_DEADLINES = dict(prepare=60., baseline=90., clock_warm=45., delay_calibration=150.,
                            scout=90., plan=90., move_entry=60., start_motion=10., finalize=30., return_rest=60.)
    terminal = frozenset(('idle', 'complete', 'cancelled', 'error', 'interrupted'))
    PUBLIC = frozenset(('quality', 'timing', 'geometry', 'cloud', 'scan_id', 'visited', 'total',
        'accepted', 'captures', 'no_return', 'rejected', 'gaps', 'point_count', 'display_point_count',
        'coverage', 'export_paths', 'timing_uncertainty_s', 'registration_reference',
        'reference_registered','reference_timing_qualified','physical_sample_timing_qualified',
        'firmware_sample_age_s','reference_timing_uncertainty_s','reference_spatial_error_bound_m','station_index','stations_complete','station_total',
        'line_progress_mm','line_samples_observed','acquisition_phase','selected_optical_roll_deg','roll_candidate_failures','line_plan_diagnostic_path'))

    def __init__(self, callbacks, *, can_start, safety_state, hold, release=None,
                 clock=time.monotonic, executor=None, deadlines=None,
                 capture_interval_s=.01, max_capture_gap_s=2., completion_reserve_s=30.):
        self.callbacks = dict(callbacks)
        self.can_start, self.safety_state = can_start, safety_state
        self.hold, self.release, self.clock = hold, release, clock
        self.deadlines = {**self.DEFAULT_DEADLINES, **(deadlines or {})}
        if (set(self.deadlines) != set(self.STAGES) or any(not math.isfinite(x) or x <= 0 for x in self.deadlines.values())
                or any(not math.isfinite(x) or x <= 0 for x in (capture_interval_s, max_capture_gap_s, completion_reserve_s))):
            raise ValueError('Finite positive stage, capture and completion deadlines are required')
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-continuous-scan')
        self._owns_executor = executor is None
        self.capture_interval_s, self.max_capture_gap_s = capture_interval_s, max_capture_gap_s
        self.completion_reserve_s = completion_reserve_s
        self.settings = ContinuousScanSettings()
        self.status, self.message, self.error = 'idle', 'Ready for one-start continuous area scanning', None
        self._generation = 0
        self._cancel_event = threading.Event()
        self._data, self._public = {}, {}
        self._future = self._capture_future = None
        self._abandoned = []
        self._started = self._phase_started = self._motion_started = None
        self._stopped = None
        self._planned_duration = self._motion_elapsed = None
        self._next_capture = self._last_qualified = None
        self._observed_motion = self._motion_complete = self._released = False
        self._closed = False

    @property
    def busy(self):
        return self.status not in self.terminal

    def _elapsed_s(self):
        if self._started is None:
            return 0.
        endpoint = self.clock() if self._stopped is None else self._stopped
        return max(0., endpoint-self._started)

    def _context(self):
        return dict(settings=asdict(self.settings), data=dict(self._data),
            cancel_event=self._cancel_event, generation=self._generation,
            started_monotonic=self._started, stopped_monotonic=self._stopped,
            workflow_elapsed_s=self._elapsed_s(), phase=self.status, outcome=self.status,
            message=self.message, error=self.error)

    @staticmethod
    def _interruption(snapshot, *, own_return_rest=False):
        shutdown = snapshot.get('shutdown') or {}
        if shutdown.get('requested') or shutdown.get('active') or snapshot.get('shutdown_requested'):
            return 'Shutdown owns return to rest', False
        if snapshot.get('cooldown_active'):
            return 'Cooldown owns return to rest', False
        if snapshot.get('fault') or snapshot.get('motion_fault'):
            return str(snapshot.get('fault') or snapshot.get('motion_fault')), False
        if not own_return_rest and (snapshot.get('closed') or snapshot.get('disable_pending')):
            return 'Arm control is disabled or disabling', False
        return None, True

    def _admission(self):
        if self._closed:
            raise RuntimeError('Continuous scan coordinator is closed')
        self._abandoned = [future for future in self._abandoned if not future.done()]
        if self._abandoned:
            raise RuntimeError('Previous cancelled scan worker is still exiting')
        missing = [name for name in (*self.STAGES, 'capture', 'motion_state') if not callable(self.callbacks.get(name))]
        if missing:
            raise RuntimeError('Continuous scan stages unavailable: '+', '.join(missing))
        reason, _ = self._interruption(self.safety_state())
        if reason:
            raise RuntimeError(reason)
        result = self.can_start()
        if result is False:
            raise RuntimeError('Arm is not ready for continuous scanning')
        if isinstance(result, dict) and not result.get('ready', True):
            raise RuntimeError(result.get('blocker') or 'Arm is not ready for continuous scanning')

    def start(self, request):
        if self.busy:
            raise RuntimeError('A continuous scan already owns the workflow')
        settings = ContinuousScanSettings.from_request(request)
        self._admission()  # Camera may be OFF; prepare owns its startup.
        self.settings = settings
        self._generation += 1
        self._cancel_event = threading.Event()
        self._data, self._public = {}, dict(visited=0, total=0, accepted=0, captures=0,
            no_return=0, rejected=0, gaps=0, point_count=0, display_point_count=0)
        self.error = None
        self._released = self._observed_motion = self._motion_complete = False
        self._planned_duration = self._motion_elapsed = None
        self._started = self.clock()
        self._stopped = None
        try:
            self._enter('prepare')
        except Exception as error:
            self._fail(error)
            raise

    def _enter(self, phase):
        self.status, self._phase_started = phase, self.clock()
        self.message = {
            'prepare': 'Preparing the existing camera owner and scan starting pose',
            'baseline': 'Qualifying the stripe, 140 mm depth and fixed optical roll',
            'clock_warm': 'Qualifying native exposure clock timing',
            'delay_calibration': 'Matching hardware exposure to kernel-received measured angles; firmware sample age unknown',
            'scout': 'Scouting the original optical XY area for a safe surface path',
            'plan': 'Validating the full continuous raster under existing physical limits',
            'move_entry': 'Moving to and verifying the validated raster entry',
            'start_motion': 'Starting the validated continuous service profile',
            'scanning': 'Registering native exposures against measured moving joint outputs',
            'finalize': 'Flushing and exporting captured surface points and measured coverage',
            'return_rest': 'Returning through the normal service path to verified rest',
        }[phase]
        if self.settings.scan_method == 'precision' and phase == 'start_motion':
            self.message = 'Capturing stabilized laser lines at 1 mm stations'
        if phase != 'scanning':
            if phase == 'start_motion' and self.clock()-self._started+self._planned_duration+self._completion_budget > self.settings.max_duration_s:
                raise ValueError('Remaining total budget cannot fit validated motion and return to rest')
            self._future = self.executor.submit(self.callbacks[phase], self._context())
        else:
            self._motion_started = self._next_capture = self._last_qualified = self.clock()

    @staticmethod
    def _poll(future):
        """Flatten injected service Futures without waiting in the robot tick."""
        if not future.done():
            return future, False, None
        value = future.result()
        if isinstance(value, Future):
            return value, False, None
        return None, True, value

    def _publish(self, result):
        if isinstance(result, dict):
            self._public.update({key: result[key] for key in self.PUBLIC if key in result})
            if result.get('message'):
                self.message = str(result['message'])

    def _phase_result(self, phase, result):
        if result is not None and not isinstance(result, dict):
            raise ValueError(f'{phase} must return an artifact dictionary')
        result = {} if result is None else result
        if result.get('error'):
            raise RuntimeError(str(result['error']))
        if result.get('ok') is False:
            raise RuntimeError(result.get('message') or f'{phase} was rejected')
        if phase == 'clock_warm' and result.get('clock_qualified') is not True and result.get('timing_qualified') is not True:
            raise ValueError('Native exposure clock timing did not qualify')
        if phase == 'delay_calibration' and result.get('timing_qualified') is not True:
            receive_ready = (result.get('reference_registered') is True
                and result.get('reference_timing_qualified') is True
                and result.get('registration_reference') == 'measured_kernel_rx'
                and result.get('physical_sample_timing_qualified') is False
                and 'firmware_sample_age_s' in result and result['firmware_sample_age_s'] is None)
            if not receive_ready:
                raise ValueError('Measured-output reference timing is unavailable or falsely qualified')
        if phase == 'plan':
            duration = result.get('duration_s')
            if result.get('validated') is not True or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('A fully validated finite continuous path duration is required')
            entry = result.get('entry_duration_s', self.deadlines['move_entry'])
            completion = result.get('completion_budget_s', self.completion_reserve_s)
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (entry, completion)):
                raise ValueError('Finite entry and completion time estimates are required')
            if self.clock()-self._started+duration+entry+self.deadlines['start_motion']+completion > self.settings.max_duration_s:
                raise ValueError('Validated scan exceeds the total time budget; physical limits cannot be widened')
            self._planned_duration = float(duration)
            self._completion_budget = float(completion)
            self._entry_duration = float(entry)
        self._data[phase] = result
        self._publish(result)
        if phase == 'start_motion':
            if result.get('stationary_protocol_complete') is True:
                self._enter('finalize')
            else:
                self._enter('scanning')
        elif phase == 'return_rest':
            if result.get('rest_verified') is not True:
                raise ValueError('Return to rest was not measured and verified')
            final = self._data.get('finalize', {})
            self.status = 'interrupted' if final.get('completion_qualified') is False else 'complete'
            self.message = (final.get('message') or 'Captured scan has unresolved measured coverage gaps') if self.status == 'interrupted' else 'Continuous scan exported; return to rest verified'
            self._release()
        else:
            self._enter(self.STAGES[self.STAGES.index(phase)+1])

    def _release(self):
        if self._released:
            return
        self._released = True
        if self._stopped is None:
            self._stopped = self.clock()
        if self.release is not None:
            try:
                self.release(self._context())
            except Exception as error:
                self.error = self.error or str(error)
        self._cancel_event.set()

    def cancel(self, *, hold=True, reason='Continuous scan cancelled; persisted points retained', _outcome='cancelled'):
        if not self.busy:
            return
        # Revoke before root-owned Hold/release, including currently running jobs.
        self._cancel_event.set()
        for future in (self._future, self._capture_future):
            if future is not None and not future.done():
                future.cancel()
                self._abandoned.append(future)
        self._future = self._capture_future = None
        self.status, self.message = _outcome, reason
        self._stopped = self.clock()
        if hold:
            try:
                blocker, healthy = self._interruption(self.safety_state())
                if healthy and not blocker:
                    self.hold()
            except Exception as error:
                self.error = self.error or str(error)
        self._release()

    def abort_for_shutdown(self):
        self.cancel(hold=False, reason='Continuous scan interrupted; shutdown owns return to rest')

    def abort_for_cooldown(self):
        self.cancel(hold=False, reason='Continuous scan interrupted; cooldown owns return to rest')

    def _fail(self, error, *, hold=True):
        self.error = str(error)
        self.cancel(hold=hold, reason='Continuous scan stopped: '+str(error), _outcome='error')

    def _capture_tick(self):
        now = self.clock()
        if self._capture_future is not None:
            try:
                self._capture_future, done, result = self._poll(self._capture_future)
                if done:
                    if result is not None and not isinstance(result, dict):
                        raise ValueError('Moving capture must return an artifact dictionary')
                    if result and result.get('qualified') is True:
                        self._last_qualified = now
                    self._publish(result)
            except RegistrationRejected as error:
                self._capture_future = None
                self._public['gaps'] = self._public.get('gaps', 0)+1
                self._public['last_capture_error'] = str(error)
            if self._capture_future is not None and now-self._capture_started > self.max_capture_gap_s:
                raise TimeoutError('Moving capture worker exceeded its bounded deadline')
        if not self._motion_complete and now-self._last_qualified > self.max_capture_gap_s:
            raise TimeoutError('No newly qualified measured exposure within the moving-capture deadline')
        if self._capture_future is None and not self._motion_complete and now >= self._next_capture:
            self._capture_started = now
            self._next_capture = max(self._next_capture+self.capture_interval_s, now)
            self._capture_future = self.executor.submit(self.callbacks['capture'], self._context())

    def tick(self, camera_state=None):
        if not self.busy:
            return
        try:
            reason, healthy = self._interruption(self.safety_state(), own_return_rest=self.status == 'return_rest')
            if reason:
                self._fail(reason, hold=healthy)
                return
            now = self.clock()
            if now-self._started > self.settings.max_duration_s:
                raise TimeoutError('Continuous workflow exceeded its total time budget')
            if self.status != 'scanning':
                stage_deadline = (180. if self.status == 'plan' and self.settings.scan_profile == 'vertical_down' else
                    600. if self.status == 'start_motion' and self.settings.scan_profile == 'vertical_down'
                    and self.settings.scan_method == 'precision' else self.deadlines[self.status])
                if now-self._phase_started > stage_deadline:
                    raise TimeoutError(f'Continuous scan {self.status} exceeded its bounded deadline')
                self._future, done, result = self._poll(self._future)
                if done:
                    self._phase_result(self.status, result)
                return
            motion = self.callbacks['motion_state'](self._context())
            if not isinstance(motion, dict):
                raise ValueError('Motion state must be a fast service snapshot')
            if motion.get('error'):
                raise RuntimeError(str(motion['error']))
            self._motion_elapsed = max(0., now-self._motion_started)
            self._observed_motion |= motion.get('active') is True
            self._motion_complete = motion.get('complete') is True
            if not self._observed_motion and not self._motion_complete and self._motion_elapsed > 2.:
                raise TimeoutError('Validated service motion never became active')
            if self._observed_motion and not motion.get('active') and not self._motion_complete:
                at_timed_endpoint = self._motion_elapsed >= self._planned_duration or motion.get('path_fraction', 0.) >= 1.
                if not at_timed_endpoint:
                    raise RuntimeError('Continuous motion ended without verified endpoint completion')
                self.message = 'Waiting for fresh quiet measured endpoint convergence'
            if self._motion_elapsed > self._planned_duration+5. and not self._motion_complete:
                raise TimeoutError('Continuous motion exceeded its validated duration and completion deadline')
            self._publish(motion)
            self._capture_tick()
            if self._motion_complete and self._capture_future is None:
                self._enter('finalize')
        except Exception as error:
            self._fail(error)

    def state(self, camera_state=None):
        ready, blocker = False, 'Continuous scan is active'
        if not self.busy:
            try:
                self._admission()
                ready, blocker = True, None
            except Exception as error:
                blocker = str(error)
        elapsed = self._elapsed_s()
        eta = None
        if self.busy and self._planned_duration is not None:
            eta = max(0., self._planned_duration-(self._motion_elapsed or 0.))+self._completion_budget
            if self.status == 'move_entry':
                eta += max(0., self._entry_duration-(self.clock()-self._phase_started))+self.deadlines['start_motion']
            elif self.status == 'start_motion':
                eta += max(0., self.deadlines['start_motion']-(self.clock()-self._phase_started))
            elif self.status in ('finalize', 'return_rest'):
                eta = max(0., self._completion_budget-(self.clock()-self._phase_started))
        return {**self._public, 'continuous': True, 'status': self.status, 'phase': self.status,
            'busy': self.busy, 'ready': ready, 'start_ready': ready, 'blocker': blocker,
            'message': self.message, 'error': self.error, 'settings': asdict(self.settings),
            'scan_profile':self.settings.scan_profile,'scan_method':self.settings.scan_method,
            'sweep_length_mm':self.settings.sweep_length_mm,
            'spacing_mm': self.settings.spacing_mm, 'standoff_mm': self.settings.standoff_mm,
            'optical_roll_deg': self.settings.optical_roll_deg, 'elapsed_s': elapsed,
            'workflow_elapsed_s': elapsed, 'stopped_monotonic': self._stopped,
            'eta_s': eta, 'validated_duration_s': self._planned_duration,
            'max_duration_s': self.settings.max_duration_s, 'resume_available': False}

    def close(self):
        self._closed = True
        self.cancel(hold=False, reason='Continuous scan coordinator closed; service owns shutdown')
        if self._owns_executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
