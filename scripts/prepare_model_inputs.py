"""Prepare fold-local clinical inputs and verify CPU data loading, not training."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.model_inputs_preflight import main

if __name__ == "__main__":
    main()
