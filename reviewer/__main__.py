"""Allow ``python -m reviewer``."""

from reviewer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
