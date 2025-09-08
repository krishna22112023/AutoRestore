import logging
import importlib
import sys
from pathlib import Path
import unittest


class TestLogging(unittest.TestCase):
    """Ensure that a timestamped log directory with info & error files is created."""

    def test_log_files_created(self):
        # Reload to trigger new timestamped folder for each test run
        if "config.logging" in sys.modules:
            importlib.reload(sys.modules["config.logging"])
        else:
            import config.logging  # noqa: F401 – side-effect creates log handlers

        logger = logging.getLogger(__name__)
        logger.info("Test info message")
        logger.error("Test error message")

        logs_root = Path(__file__).resolve().parents[1] / "logs"
        subdirs = [d for d in logs_root.iterdir() if d.is_dir()]
        self.assertTrue(subdirs, "No log directories created under 'logs/'")
        latest = max(subdirs, key=lambda p: p.stat().st_mtime)
        self.assertTrue((latest / "info.json").exists(), "info.json missing in latest log dir")
        self.assertTrue((latest / "error.json").exists(), "error.json missing in latest log dir")


if __name__ == "__main__":
    unittest.main()
