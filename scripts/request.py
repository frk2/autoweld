"""Submit a local request to the running application, without another hardware connection."""

import json
import time
import uuid
from pathlib import Path

import tyro


def main(payload: str, runtime: str = "runtime/sim", timeout: float = 20):
    root = Path(runtime)
    req = json.loads(payload)
    name = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    path = root / "requests" / f"{name}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(req))
    temp.replace(path)
    response = root / "responses" / f"{name}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if response.exists():
            print(response.read_text())
            return
        time.sleep(0.05)
    raise TimeoutError("No response; check the app log and status.json. Do not resend a motion blindly.")


if __name__ == "__main__":
    tyro.cli(main)
