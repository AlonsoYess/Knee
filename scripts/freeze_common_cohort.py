"""Freeze the approved common cohort using a private configuration."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.common_cohort import main

if __name__ == "__main__":
    main()
