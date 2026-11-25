"""Exercise startup ownership without importing device/model/native libraries."""

import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace


def test_application_releases_startup_reference_only_after_controller_replaces_owner():
    source = ast.parse(Path('autoweld/backend.py').read_text())
    run = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run_application')
    index = next(index for index, node in enumerate(run.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'service'
            for target in node.targets))
    # Execute the actual ownership handoff rather than recreating it in a test.
    handoff = ast.Module(body=run.body[index:index+2], type_ignores=[])

    class Owner:
        pass

    class Service:
        def __init__(self, robot, *_args, **_kwargs):
            self.ctrl = SimpleNamespace(robot=robot)

    original = Owner()
    original_ref = weakref.ref(original)
    namespace = dict(robot=original, RobotService=Service, model_path=None, runtime=None, live=False,
        connect_robot=None, start_disabled=True, camera=None, host_guard=None,
        recovery_reader=None, disabled_reader=None)
    del original
    exec(compile(handoff, '<actual-startup-handoff>', 'exec'), namespace)
    assert 'robot' not in namespace
    assert original_ref() is namespace['service'].ctrl.robot
    # Replacement is performed by the sole controller. The long-lived startup
    # frame must no longer retain the old display or a previously closed owner.
    replacement = Owner()
    namespace['service'].ctrl.robot = replacement
    gc.collect(0)
    assert original_ref() is None
    assert namespace['service'].ctrl.robot is replacement
