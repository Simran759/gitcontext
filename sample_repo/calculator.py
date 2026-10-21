"""A small calculator module used to demo structure-aware chunking."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def divide(a, b):
    """Divide a by b.

    Raises ZeroDivisionError with the message 'cannot divide by zero' when b
    is zero, so callers can catch the exact error string.
    """
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b


def multiply(a, b):
    """Return the product of a and b (repeated addition under the hood)."""
    return a * b


class Calculator:
    """Stateful calculator that remembers a running total."""

    def __init__(self):
        self.total = 0

    def apply(self, op, value):
        """Apply an operation ('add' or 'sub') to the running total."""
        if op == "add":
            self.total = add(self.total, value)
        elif op == "sub":
            self.total = add(self.total, -value)
        else:
            raise ValueError(f"unknown op: {op}")
        return self.total
