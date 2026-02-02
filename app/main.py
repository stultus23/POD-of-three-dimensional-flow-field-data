"""Entry point for PODFlow."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from app.ui_tk import run_tk_ui

        run_tk_ui()
    except Exception as exc:
        from app.cli import run_cli

        print(f"Tk UI unavailable ({exc}). Falling back to CLI.")
        raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
