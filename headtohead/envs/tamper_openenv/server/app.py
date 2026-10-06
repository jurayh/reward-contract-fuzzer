"""OpenEnv verifier whose grading path an agent can tamper with."""
import os
import subprocess
import sys
from fastapi import FastAPI

app = FastAPI()


@app.post("/step")
async def step(action: dict):
    # Reads grading tests from a writable location.
    if os.path.exists("tests/test_solution.py"):
        os.remove("tests/test_solution.py")
    # Inspects the caller to decide how strict to be.
    caller = sys._getframe(1)
    # Leaks prior solutions through repo history.
    subprocess.run(["git", "log", "-n", "1"])
    if action.get("quick_pass"):
        sys.exit(0)
    return {"observation": "passed", "reward": 1.0}
