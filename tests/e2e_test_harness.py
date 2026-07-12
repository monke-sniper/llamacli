"""Process-level E2E test harness for llamacli.

Spawns the real CLI in a subprocess and interacts via stdin/stdout.
This tests the actual binary, not imported Python functions.
"""
import io
import os
import re
import subprocess
import sys
import threading
import time
from typing import Optional


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub('', text)


class CliRunner:
    """Spawns llamacli as a real subprocess and provides send/expect helpers."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.proc: Optional[subprocess.Popen] = None
        self._output_buffer = io.StringIO()
        self._stdout_reader: Optional[threading.Thread] = None
        self._read_lock = threading.Lock()
        self._stop_reading = threading.Event()

    def start(self, args: Optional[list] = None, timeout: float = 30):
        """Spawn llamacli with an isolated workspace.

        Uses `python -m llamacli.cli` to call the Typer app directly,
        bypassing `entry()` venv forwarding logic. This keeps tests fast
        and deterministic while still testing real subprocess stdin/stdout.
        """
        env = {**os.environ, "LLAMACLII_WORKSPACE": self.workspace_dir}
        cmd = [sys.executable, "-m", "llamacli.cli"] + (args or [])

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        self._stop_reading.clear()
        self._stdout_reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_reader.start()

        # Give the process a moment to initialize
        time.sleep(0.5)

    def _read_stdout(self):
        """Background thread that drains stdout into the buffer."""
        if not self.proc or not self.proc.stdout:
            return
        while not self._stop_reading.is_set():
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                with self._read_lock:
                    self._output_buffer.write(line)
            except Exception:
                break

    def _get_output(self) -> str:
        """Return the current buffered output."""
        with self._read_lock:
            return self._output_buffer.getvalue()

    def send(self, text: str):
        """Send a line to stdin (adds newline automatically)."""
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Process not started")
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def send_raw(self, text: str):
        """Send raw text without adding a newline."""
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Process not started")
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def wait_for(self, marker: str, timeout: float = 10) -> str:
        """Wait until marker appears in output, then return all output.

        Args:
            marker: Substring to wait for.
            timeout: Max seconds to wait.

        Returns:
            All output seen so far.

        Raises:
            TimeoutError: If marker doesn't appear within timeout.
        """
        deadline = time.time() + timeout
        stripped_marker = strip_ansi(marker)
        while time.time() < deadline:
            output = self._get_output()
            clean = strip_ansi(output)
            if stripped_marker.lower() in clean.lower():
                return output
            time.sleep(0.1)
        output = self._get_output()
        raise TimeoutError(
            f"Timeout waiting for marker {marker!r}\n"
            f"Captured output (last 500 chars):\n{output[-500:]}"
        )

    def read_available(self, timeout: float = 2) -> str:
        """Read whatever output is currently available.

        Args:
            timeout: Seconds to wait for new output.
        """
        time.sleep(timeout)
        return self._get_output()

    def assert_contains(self, text: str, timeout: float = 5):
        """Assert that output contains text (waits up to timeout)."""
        self.wait_for(text, timeout=timeout)

    def assert_exit_code(self, expected: int = 0, timeout: float = 30):
        """Wait for process to exit and assert return code.

        Args:
            expected: Expected exit code.
            timeout: Max seconds to wait for process to finish.
        """
        if not self.proc:
            raise RuntimeError("Process not started")

        # Close stdin so the process can properly finish
        if self.proc.stdin:
            self.proc.stdin.close()

        try:
            returncode = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            returncode = -1

        if returncode != expected:
            output = self._get_output()
            raise AssertionError(
                f"Expected exit code {expected}, got {returncode}\n"
                f"Output (last 800 chars): {output[-800:]}"
            )

    def get_output(self) -> str:
        """Return all captured output."""
        return self._get_output()

    def get_clean_output(self) -> str:
        """Return all captured output with ANSI codes stripped."""
        return strip_ansi(self._get_output())

    def close(self, timeout: float = 5):
        """Terminate the process and clean up resources."""
        if not self.proc:
            return

        self._stop_reading.set()

        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)

        if self._stdout_reader and self._stdout_reader.is_alive():
            self._stdout_reader.join(timeout=2)

        self._output_buffer.close()

    def is_running(self) -> bool:
        """Return True if the subprocess is still running."""
        return self.proc is not None and self.proc.poll() is None
