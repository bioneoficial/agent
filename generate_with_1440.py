# File: generate_with_1440.py

def add(x, y):
    """
    Returns the sum of x and y.

    Args:
        x (int): The first number.
        y (int): The second number.

    Returns:
        int: The sum of x and y.
    """
    return x + y


def subtract(x, y):
    """
    Returns the difference between x and y.

    Args:
        x (int): The first number.
        y (int): The second number.

    Returns:
        int: The difference between x and y.
    """
    return x - y


def multiply(x, y):
    """
    Returns the product of x and y.

    Args:
        x (int): The first number.
        y (int): The second number.

    Returns:
        int: The product of x and y.
    """
    return x * y


def divide(x, y):
    """
    Returns the quotient of x divided by y.

    Args:
        x (int): The dividend.
        y (int): The divisor.

    Returns:
        float: The quotient of x divided by y.

    Raises:
        ZeroDivisionError: If y is zero.
    """
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return x / y


def main():
    print("Calculator Functions")
    while True:
        print("\nOptions:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Quit")
        
        choice = input("Choose an option: ")
        
        if choice == "5":
            break
        
        elif choice in ["1", "2", "3", "4"]:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))
            
            if choice == "1":
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == "2":
                print(f"{num1} - {num2} = {subtract(num1, num2)}")
            elif choice == "3":
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == "4":
                try:
                    print(f"{num1} / {num2} = {divide(num1, num2)}")
                except ZeroDivisionError as e:
                    print(str(e))
        else:
            print("Invalid option. Please choose a valid option.")


if __name__ == "__main__":
    main()