import pytest
# Assuming the functions are defined in main.py
# If you put the functions in a module named 'utils', you would use: from utils import add, subtract, multiply, divide

# For simplicity, we assume the functions are available in the same scope/module
# so we will copy the necessary functions here for the test file to be self-contained
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b

# --- Test Cases ---

def test_add_positive_numbers():
    """Test basic addition of two positive numbers."""
    assert add(10, 5) == 15

def test_add_negative_numbers():
    """Test addition with negative numbers."""
    assert add(-10, -5) == -15

def test_add_mixed_numbers():
    """Test addition of positive and negative numbers."""
    assert add(10, -5) == 5

def test_subtract_positive_numbers():
    """Test basic subtraction of two positive numbers."""
    assert subtract(10, 5) == 5

def test_subtract_reverse_order():
    """Test subtraction where the result is negative."""
    assert subtract(5, 10) == -5

def test_multiply_positive_numbers():
    """Test basic multiplication."""
    assert multiply(10, 5) == 50

def test_multiply_by_zero():
    """Test multiplication by zero."""
    assert multiply(100, 0) == 0

def test_divide_positive_numbers():
    """Test basic division."""
    assert divide(10, 5) == 2.0

def test_divide_floating_point():
    """Test division resulting in a float."""
    assert divide(10, 4) == 2.5

def test_divide_by_zero():
    """Test that dividing by zero raises a ValueError."""
    with pytest.raises(ValueError, match="Denominator cannot be zero."):
        divide(10, 0)

def test_divide_zero_numerator():
    """Test dividing zero by a non-zero number."""
    assert divide(0, 5) == 0.0

def test_divide_zero_by_zero():
    """Test dividing zero by zero (should still raise ValueError per implementation)."""
    # Since our function treats b==0 as an error, this test confirms that behavior.
    with pytest.raises(ValueError, match="Denominator cannot be zero."):
        divide(0, 0)
