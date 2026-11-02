#!/usr/bin/env python3

import sys


def add(a, b):
    """Return the sum of a and b."""
    return a + b


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add.py <a> <b>")
        sys.exit(1)

    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
    except ValueError:
        print("Both arguments must be numbers.")
        sys.exit(1)

    result = add(a, b)
    # Print without trailing .0 when the result is an integer value
    if isinstance(result, float) and result.is_integer():
        print(int(result))
    else:
        print(result)
