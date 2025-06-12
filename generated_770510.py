# Importing required libraries for math operations
import math

def add(x, y):
    """This function adds two numbers"""
    return x + y

def subtract(x, y):
    """This function subtracts one number from another"""
    return x - y

def multiply(x, y):
    """This function multiplies two numbers"""
    return x * y

def divide(x, y):
    """This function divides one number by another"""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

# Main calculator function
def calculator():
    print("Simple Calculator Program")

    while True:
        # Display menu for user to select operation
        print("\nChoose an operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice in ['1', '2', '3', '4']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))

            # Based on user's choice, select operation and perform
            if choice == '1':
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == '2':
                try:
                    print(f"{num1} - {num2} = {subtract(num1, num2)}")
                except ValueError as e:
                    print(str(e))
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == '4':
                try:
                    print(f"{num1} / {num2} = {divide(num1, num2)}")
                except ValueError as e:
                    print(str(e))
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please choose between 1 and 5.")

if __name__ == "__main__":
    calculator()