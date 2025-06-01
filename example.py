#!/usr/bin/env python3
"""
Example Python module to test commit message generation
"""

def add(a, b):
    """Add two numbers and return the result"""
    return a + b

def subtract(a, b):
    """Subtract b from a and return the result"""
    return a - b

def multiply(a, b):
    """Multiply two numbers and return the result"""
    return a * b

def divide(a, b):
    """Divide a by b and return the result
    
    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    print(f"10 + 5 = {add(10, 5)}")
    print(f"10 - 5 = {subtract(10, 5)}")
    print(f"10 * 5 = {multiply(10, 5)}")
    print(f"10 / 5 = {divide(10, 5)}")
