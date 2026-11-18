"""Compatibility entry point for the approved pT-only variant."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("prepare_pt_only_inputs.py")), run_name="__main__")
