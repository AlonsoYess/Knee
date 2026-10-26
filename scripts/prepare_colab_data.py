"""Prepare fixed grouped partitions and a private Colab data handoff without fitting."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.colab_preparation import main

if __name__ == "__main__":
    main()
