"""Testing suite including integration and unit tests."""
from pathlib import Path

TEST_DATA = Path(__file__).parent.parent / "test_data"
RESOURCES = Path(__file__).parent.parent / "src" / "resources"
INPUT_FILES = list(file for file in (TEST_DATA / "inputs").iterdir())
