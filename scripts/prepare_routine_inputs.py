"""Apply the fixed model-input transformation using a private routine config."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from knee.routine_model_inputs import main

if __name__ == "__main__":
    main()
