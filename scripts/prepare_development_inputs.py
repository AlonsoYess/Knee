"""Run the approved development-only image-input preparation in its frozen Python."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from knee.development_model_inputs import main

if __name__ == "__main__":
    main()
