"""Compatibility wrapper for the parser migrated into :mod:`src`.

The experiment tests and scripts can continue importing ``pdf_to_markdown``
while production code uses ``src.pdf_parser`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import pdf_parser as _implementation


globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)

# Make unittest.mock patch targets behave exactly as they did before the move.
sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation._cli())
