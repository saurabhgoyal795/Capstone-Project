"""Allow ``python -m complaint_processor``."""

import sys

from complaint_processor.cli import main

if __name__ == "__main__":
    sys.exit(main())
