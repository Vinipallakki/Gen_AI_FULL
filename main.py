def add(a,b):
    return a + b

def subtrct(a,b):
    return a - b

def multiply(a,b):
    return a * b

def diide(a,b):
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b    

add(10, 5)
subtrct(10, 5)
multiply(10, 5)
diide(10, 5)
