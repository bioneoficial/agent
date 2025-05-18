# Script para verificar se um número é par ou ímpar

try:
    num = int(input("Digite um número: "))
    
    if num % 2 == 0:
        print(f"O número {num} é par.")
    else:
        print(f"O número {num} é ímpar.")
        
except ValueError:
    print("Por favor, digite apenas números inteiros.")