"""Allow `python -m samplesize ...` to dispatch to the CLI."""
from samplesize.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
