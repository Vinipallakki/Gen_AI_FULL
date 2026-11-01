def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python add.py <a> <b>")
        raise SystemExit(1)

    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
    except ValueError:
        print("Please provide numeric values.")
        raise SystemExit(1)

    print(add(a, b))
