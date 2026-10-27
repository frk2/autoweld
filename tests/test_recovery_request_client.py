"""The agent request helper timestamps explicit recovery intent before queueing."""
import json

import pytest

from scripts import request


@pytest.mark.parametrize('action', ['recover_control', 'sync'])
def test_recovery_submission_time_is_written_before_waiting(tmp_path, monkeypatch, capsys, action):
    (tmp_path / 'requests').mkdir()
    (tmp_path / 'responses').mkdir()
    monkeypatch.setattr(request.time, 'time', lambda: 1000.)
    received = []

    def respond(_seconds):
        paths = list((tmp_path / 'requests').glob('*.json'))
        assert len(paths) == 1
        received.append(json.loads(paths[0].read_text()))
        (tmp_path / 'responses' / paths[0].name).write_text('{"ok": true}')

    monkeypatch.setattr(request.time, 'sleep', respond)
    request.main(json.dumps({'action': action}), str(tmp_path))
    assert json.loads(capsys.readouterr().out) == {'ok': True}
    assert received == [{'action': action, **({'requested_at': 1000.} if action == 'recover_control' else {})}]
