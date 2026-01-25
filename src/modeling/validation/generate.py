#!/usr/bin/env python

import sys

from generate_peptides import main as generate_main
from latent_viz import main as latent_main


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "latent-viz":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return latent_main()
    return generate_main()


if __name__ == "__main__":
    raise SystemExit(main())
