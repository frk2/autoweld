"""Deterministic coordinator checks: no camera/SDK/model/browser is created."""

from concurrent.futures import Future

import pytest

from autoweld.continuous_scanning import ContinuousScanCoordinator, ContinuousScanSettings
from autoweld.motion_registration import RegistrationRejected


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class Workflow:
    def __init__(self, **options):
        self.clock = Clock()
        self.events, self.holds, self.releases = [], [], []
        self.safety, self.motion = {"closed": False}, {"active": True, "complete": False}
        self.captures = 0
        self.callbacks = {phase: self.stage(phase) for phase in ContinuousScanCoordinator.STAGES}
        self.callbacks.update(capture=self.capture, motion_state=lambda _ctx: self.motion)
        self.scan = ContinuousScanCoordinator(
            self.callbacks,
            can_start=lambda: None,
            safety_state=lambda: self.safety,
            hold=lambda: self.holds.append(True),
            release=lambda ctx: self.releases.append(ctx),
            clock=self.clock,
            executor=ImmediateExecutor(),
            **options,
        )

    def stage(self, phase):
        def callback(ctx):
            assert not ctx["cancel_event"].is_set()
            self.events.append((phase, ctx))
            if phase in ("clock_warm", "delay_calibration"):
                return {"timing_qualified": True}
            if phase == "plan":
                return {"validated": True, "duration_s": 642.6, "entry_pose": "validated"}
            if phase == "return_rest":
                return {"rest_verified": True}
            return {}

        return callback

    def capture(self, ctx):
        assert not ctx["cancel_event"].is_set()
        self.captures += 1
        return dict(
            qualified=True,
            captures=self.captures,
            accepted=self.captures,
            visited=self.captures * 20,
            total=10201,
            timing_uncertainty_s=0.0007,
        )

    def until(self, phase):
        for _ in range(30):
            if self.scan.status == phase:
                return
            self.scan.tick()
        assert self.scan.status == phase


def test_one_start_owns_all_stages_camera_off_motion_capture_export_and_rest():
    w = Workflow()
    assert w.scan.state()["ready"]  # No stream/device is required for admission.
    w.scan.start({"continuous": True})
    w.until("scanning")
    assert [phase for phase, _ in w.events] == list(w.scan.STAGES[:8])
    assert w.scan.settings == ContinuousScanSettings()
    assert w.scan._data["plan"]["entry_pose"] == "validated"
    w.scan.tick()  # Start a capture; no stationary gate during validated motion.
    w.clock.advance(0.1)
    w.scan.tick()
    assert w.scan.state()["accepted"] == 1
    assert w.scan.state()["timing_uncertainty_s"] == 0.0007
    assert w.scan.state()["validated_duration_s"] == 642.6
    w.motion = {"active": False, "complete": True}
    w.until("complete")
    assert [phase for phase, _ in w.events][-2:] == ["finalize", "return_rest"]
    assert len(w.releases) == 1 and not w.holds
    assert not w.scan.busy and not w.scan.state()["resume_available"]


def test_prepare_callback_is_submitted_once_not_reissued_by_ticks():
    pending = Future()
    w = Workflow()
    w.scan.callbacks["prepare"] = lambda ctx: pending
    w.scan.start({})
    for _ in range(50):
        w.scan.tick()
    assert w.scan.status == "prepare"
    pending.set_result({"camera_ready": True})
    w.scan.tick()
    assert w.scan._data["prepare"]["camera_ready"]
    assert w.scan.status == "baseline"


@pytest.mark.parametrize("phase", ContinuousScanCoordinator.STAGES)
def test_every_stage_has_bounded_deadline_and_cancels_stale_work(phase):
    w = Workflow()
    pending = Future()
    w.scan.callbacks[phase] = lambda _ctx: pending
    w.scan.start({})
    if phase in ("finalize", "return_rest"):
        w.until("scanning")
        w.motion = {"complete": True, "active": False}
    w.until(phase)
    w.clock.advance(w.scan.deadlines[phase] + 0.001)
    w.scan.tick()
    assert w.scan.status == "error"
    assert "deadline" in w.scan.error
    assert w.scan._cancel_event.is_set()
    assert len(w.holds) == len(w.releases) == 1


@pytest.mark.parametrize("phase", ContinuousScanCoordinator.STAGES + ("scanning",))
@pytest.mark.parametrize(
    "safety",
    [
        dict(shutdown={"requested": True}),
        dict(cooldown_active=True),
        dict(fault="hardware fault"),
        dict(motion_fault="tracking fault"),
    ],
)
def test_shutdown_cooldown_and_fault_release_without_hold_in_every_phase(phase, safety):
    w = Workflow()
    w.scan.start({})
    if phase in ("finalize", "return_rest"):
        w.until("scanning")
        w.motion = {"complete": True, "active": False}
    w.until(phase)
    w.safety.update(safety)
    w.scan.tick()
    assert not w.scan.busy
    assert not w.holds and len(w.releases) == 1
    assert w.scan._cancel_event.is_set()


def test_operator_cancel_is_idempotent_single_hold_and_no_resume():
    w = Workflow()
    w.scan.start({})
    w.until("scanning")
    w.scan.cancel()
    w.scan.cancel()
    w.scan.tick()
    assert w.scan.status == "cancelled"
    assert len(w.holds) == len(w.releases) == 1
    assert not w.scan.state()["resume_available"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizontal_mm", 100.01),
        ("vertical_mm", 0),
        ("horizontal_mm", float("nan")),
        ("vertical_mm", True),
        ("optical_roll_deg", 181.0),
        ("feed_mm_s", 30.01),
        ("feed_mm_s", 0.0),
        ("max_duration_s", 901),
        ("start_pose", "other"),
    ],
)
def test_invalid_request_fails_before_any_work(field, value):
    w = Workflow()
    with pytest.raises(ValueError):
        w.scan.start({field: value})
    assert not w.events and not w.holds and not w.scan.busy


@pytest.mark.parametrize("phase", ["clock_warm", "delay_calibration"])
def test_unqualified_timing_never_reaches_raster(phase):
    w = Workflow()
    w.scan.callbacks[phase] = lambda _ctx: {"timing_qualified": False}
    w.scan.start({})
    w.until("error")
    assert not any(name in ("plan", "start_motion") for name, _ in w.events)
    assert "timing" in w.scan.error


@pytest.mark.parametrize(
    "plan",
    [
        dict(validated=False, duration_s=10.0),
        dict(validated=True, duration_s=901.0),
        dict(validated=True, duration_s=float("nan")),
        dict(validated=True, duration_s=True),
    ],
)
def test_unvalidated_or_over_budget_plan_never_moves_entry(plan):
    w = Workflow()
    w.scan.callbacks["plan"] = lambda _ctx: plan
    w.scan.start({})
    w.until("error")
    assert not any(name == "move_entry" for name, _ in w.events)


def test_budget_uses_elapsed_preparation_and_actual_validated_duration():
    w = Workflow()
    w.scan.start({})
    w.until("plan")
    w.clock.advance(240.0)
    # Avoid phase deadline masking the overall estimated completion rejection.
    w.scan._phase_started = w.clock()
    w.scan.tick()
    assert w.scan.status == "error"
    assert "total time budget" in w.scan.error


def test_good_no_return_is_qualified_observation_without_surface_points():
    w = Workflow()
    w.scan.callbacks["capture"] = lambda _ctx: dict(qualified=True, no_return=1, accepted=0, point_count=0)
    w.scan.start({})
    w.until("scanning")
    for _ in range(6):
        w.scan.tick()
        w.clock.advance(0.5)
    assert w.scan.status == "scanning"
    assert w.scan.state()["no_return"] == 1 and w.scan.state()["point_count"] == 0


def test_one_capture_inflight_and_registration_rejects_are_counted_then_bound():
    w = Workflow()
    calls = []

    def reject(ctx):
        calls.append(ctx)
        raise RegistrationRejected("Missing measured right bracket")

    w.scan.callbacks["capture"] = reject
    w.scan.start({})
    w.until("scanning")
    w.scan.tick()
    w.scan.tick()
    assert w.scan.state()["gaps"] == 1
    w.clock.advance(2.01)
    w.scan.tick()
    assert w.scan.status == "error"
    assert "qualified measured exposure" in w.scan.error
    assert len(w.holds) == 1


def test_pending_capture_cannot_backlog_and_has_deadline():
    w = Workflow()
    pending = Future()
    calls = []
    w.scan.callbacks["capture"] = lambda ctx: calls.append(ctx) or pending
    w.scan.start({})
    w.until("scanning")
    for _ in range(20):
        w.scan.tick()
        w.clock.advance(0.05)
    assert len(calls) == 1
    w.clock.advance(2.0)
    w.scan.tick()
    assert w.scan.status == "error" and pending.cancelled()


def test_motion_cannot_finish_by_elapsed_time_alone_or_disappear():
    w = Workflow()
    w.scan.start({})
    w.until("scanning")
    w.scan.tick()
    w.motion = {"active": False, "complete": False}
    w.scan.tick()
    assert w.scan.status == "error"
    assert "without verified endpoint" in w.scan.error
    assert not any(name == "finalize" for name, _ in w.events)


def test_return_rest_requires_measured_verification():
    w = Workflow()
    w.scan.callbacks["return_rest"] = lambda ctx: {"rest_verified": False}
    w.scan.start({})
    w.until("scanning")
    w.motion = {"complete": True, "active": False}
    w.until("error")
    assert "rest" in w.scan.error


def test_no_stale_worker_results_or_new_scan_until_old_callback_exits():
    w = Workflow()
    running = Future()
    running.set_running_or_notify_cancel()
    w.scan.callbacks["prepare"] = lambda ctx: running
    w.scan.start({})
    w.scan.tick()  # Unwrap outer Future to the running stage Future.
    generation = w.scan._generation
    w.scan.cancel()
    with pytest.raises(RuntimeError, match="still exiting"):
        w.scan.start({})
    running.set_result({"message": "stale result"})
    w.scan.tick()
    assert w.scan.message != "stale result"
    w.scan.callbacks["prepare"] = w.stage("prepare")
    w.scan.start({})
    assert w.scan._generation == generation + 1
    assert not w.scan._cancel_event.is_set()


def test_close_never_holds_over_service_shutdown_and_prevents_new_work():
    w = Workflow()
    w.scan.start({})
    w.scan.close()
    assert not w.holds and len(w.releases) == 1
    with pytest.raises(RuntimeError, match="closed"):
        w.scan.start({})


def test_failure_outcome_is_published_before_ownership_release():
    w = Workflow()
    w.scan.callbacks["prepare"] = lambda _ctx: {"error": "No matching optical calibration"}
    w.scan.start({})
    w.scan.tick()
    assert w.releases[0]["outcome"] == "error"
    assert "optical" in w.releases[0]["error"]


def test_safety_snapshot_failure_does_not_escape_tick_or_interrupt_native_writer():
    w = Workflow()
    w.scan.start({})

    def broken_snapshot():
        raise RuntimeError("Unavailable safety snapshot")

    w.scan.safety_state = broken_snapshot
    w.scan.tick()
    assert w.scan.status == "error"
    assert len(w.releases) == 1 and not w.holds


def test_entry_delay_cannot_launch_motion_that_will_exceed_remaining_budget():
    w = Workflow()
    pending = Future()
    w.scan.callbacks["move_entry"] = lambda _ctx: pending
    w.scan.start({})
    w.until("move_entry")
    w.scan.tick()  # Inner future.
    w.clock.advance(240.0)
    w.scan._phase_started = w.clock()  # Isolate final launch budget from entry timeout.
    pending.set_result({})
    w.scan.tick()
    assert w.scan.status == "error"
    assert not any(name == "start_motion" for name, _ in w.events)


def test_reported_eta_includes_validated_raster_entry_and_finish_budget():
    w = Workflow()
    w.scan.callbacks["plan"] = lambda _ctx: dict(
        validated=True, duration_s=642.6, entry_duration_s=12.0, completion_budget_s=22.0
    )
    w.scan.start({})
    w.until("move_entry")
    assert w.scan.state()["eta_s"] == pytest.approx(642.6 + 12.0 + 10.0 + 22.0)


def test_own_normal_disable_at_rest_can_finish_with_closed_feedback():
    w = Workflow()
    pending = Future()
    w.scan.callbacks["return_rest"] = lambda _ctx: pending
    w.scan.start({})
    w.until("scanning")
    w.motion = {"complete": True, "active": False}
    w.until("return_rest")
    w.safety.update(disable_pending=True)
    w.scan.tick()
    assert w.scan.status == "return_rest"
    w.safety.update(disable_pending=False, closed=True)
    pending.set_result({"rest_verified": True})
    w.scan.tick()
    assert w.scan.status == "complete" and not w.holds


def test_external_shutdown_still_preempts_own_disable_at_rest():
    w = Workflow()
    w.scan.start({})
    w.until("scanning")
    w.motion = {"complete": True, "active": False}
    w.until("return_rest")
    w.safety.update(disable_pending=True, shutdown={"requested": True})
    w.scan.tick()
    assert w.scan.status == "error" and not w.holds


def test_timed_endpoint_waits_for_fresh_quiet_measurements_before_completion():
    w = Workflow()
    w.scan.start({})
    w.until('scanning')
    w.scan.tick()
    w.clock.advance(642.6)
    w.scan._last_qualified = w.clock()
    w.motion = {'active':False,'complete':False,'path_fraction':1.}
    w.scan.tick()
    assert w.scan.status == 'scanning'
    assert 'quiet measured endpoint' in w.scan.message
    w.clock.advance(.2)
    w.motion = {'active':False,'complete':True,'path_fraction':1.}
    w.until('complete')
    assert not w.holds


def test_exported_unresolved_coverage_returns_rest_without_claiming_complete():
    w = Workflow()
    w.scan.callbacks['finalize'] = lambda _ctx:dict(completion_qualified=False,
        message='Captured points exported with unresolved measured coverage gaps')
    w.scan.start({})
    w.until('scanning')
    w.motion = {'active':False,'complete':True}
    w.until('interrupted')
    assert w.events[-1][0] == 'return_rest'
    assert 'unresolved' in w.scan.message
    assert w.releases[-1]['outcome'] == 'interrupted'


@pytest.mark.parametrize('outcome',['cancelled','error','complete','interrupted'])
def test_terminal_workflow_elapsed_and_eta_do_not_grow_after_stop(outcome):
    w = Workflow()
    if outcome == 'error':
        w.scan.callbacks['prepare'] = lambda _ctx:{'error':'Native timing qualification failed'}
    elif outcome == 'interrupted':
        w.scan.callbacks['finalize'] = lambda _ctx:{'completion_qualified':False,'message':'Coverage gaps'}
    w.scan.start({})
    w.clock.advance(7.)
    if outcome == 'cancelled':
        w.scan.cancel()
    elif outcome == 'error':
        w.scan.tick()
    else:
        w.until('scanning')
        w.motion = {'active':False,'complete':True}
        w.until(outcome)
    stopped = w.scan.state()
    assert stopped['status'] == outcome
    assert stopped['elapsed_s'] == stopped['workflow_elapsed_s'] == 7.
    assert stopped['stopped_monotonic'] == w.clock()
    assert w.releases[-1]['workflow_elapsed_s'] == 7.
    w.clock.advance(1000.)
    w.scan.tick()
    later = w.scan.state()
    assert later['elapsed_s'] == stopped['elapsed_s']
    assert later['stopped_monotonic'] == stopped['stopped_monotonic']
    assert later['eta_s'] == stopped['eta_s']


def test_new_start_resets_previous_terminal_elapsed_clock():
    w = Workflow()
    w.scan.start({})
    w.clock.advance(7.)
    w.scan.cancel()
    w.clock.advance(100.)
    w.scan.start({})
    state = w.scan.state()
    assert state['elapsed_s'] == 0.
    assert state['stopped_monotonic'] is None
    w.clock.advance(3.)
    assert w.scan.state()['elapsed_s'] == 3.


def test_hardware_clock_and_receive_reference_readiness_are_not_physical_qualification():
    w = Workflow()
    w.scan.callbacks['clock_warm'] = lambda _ctx:dict(clock_qualified=True)
    w.scan.callbacks['delay_calibration'] = lambda _ctx:dict(reference_registered=True,
        reference_timing_qualified=True,registration_reference='measured_kernel_rx',
        physical_sample_timing_qualified=False,firmware_sample_age_s=None)
    w.scan.start({})
    w.until('scanning')
    state = w.scan.state()
    assert state['reference_registered'] and state['reference_timing_qualified']
    assert state['physical_sample_timing_qualified'] is False
    assert state['firmware_sample_age_s'] is None
    assert w.scan.settings.empirical_timing_diagnostic is False


@pytest.mark.parametrize('incorrect',[dict(physical_sample_timing_qualified=True),
    dict(firmware_sample_age_s=0.),dict(reference_timing_qualified=False)])
def test_reference_setup_cannot_silently_claim_unknown_firmware_timing_is_known(incorrect):
    w = Workflow()
    result = dict(reference_registered=True,reference_timing_qualified=True,
        registration_reference='measured_kernel_rx',physical_sample_timing_qualified=False,firmware_sample_age_s=None)
    result.update(incorrect)
    w.scan.callbacks['delay_calibration'] = lambda _ctx:result
    w.scan.start({})
    w.until('error')
    assert 'reference timing' in w.scan.error
    assert not any(phase == 'plan' for phase,_ in w.events)
