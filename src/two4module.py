"""Entry point for the casting random-forest analysis."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.casting_rf import main


if __name__ == "__main__":
    main()
