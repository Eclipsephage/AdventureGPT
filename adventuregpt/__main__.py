"""
AdventureGPT module entrypoint.

Usage:
    python -m adventuregpt [--help]

This delegates to `adventuregpt.cli`.
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
