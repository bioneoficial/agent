#!/usr/bin/env python3

class Calculadora:
    def __init__(self):
        """
        Inicializa a classe calculadora.
        
        Args:
            None
        
        Returns:
            None
        """

    @staticmethod
    def somar(num1: float, num2: float) -> float:
        """
        Realiza a operação de soma entre dois números.
        
        Args:
            num1 (float): Primeiro número.
            num2 (float): Segundo número.
        
        Returns:
            float: Resultado da operação de soma.
        """
        return num1 + num2

    @staticmethod
    def subtrair(num1: float, num2: float) -> float:
        """
        Realiza a operação de subtração entre dois números.
        
        Args:
            num1 (float): Primeiro número.
            num2 (float): Segundo número.
        
        Returns:
            float: Resultado da operação de subtração.
        """
        return num1 - num2

    @staticmethod
    def multiplicar(num1: float, num2: float) -> float:
        """
        Realiza a operação de multiplicar dois números.
        
        Args:
            num1 (float): Primeiro número.
            num2 (float): Segundo número.
        
        Returns:
            float: Resultado da operação de multiplicação.
        """
        return num1 * num2
    
    @staticmethod
    def dividir(num1: float, num2: float) -> float:
        """
        Realiza a operação de divisão entre dois números.
        
        Args:
            num1 (float): Numerador.
            num2 (float): Denominador.
        
        Returns:
            float: Resultado da operação de divisão.
            
        Raises:
            ZeroDivisionError: Se o denominador for zero.
        """
        if num2 == 0:
            raise ZeroDivisionError("Não é possível dividir por zero.")
        return num1 / num2

    @staticmethod
    def potencia(num1: float, num2: float) -> float:
        """
        Realiza a operação de potência entre dois números.
        
        Args:
            num1 (float): Base.
            num2 (float): Exponente.
        
        Returns:
            float: Resultado da operação de potência.
            
        Raises:
            ValueError: Se o expoente for negativo.
        """
        if num2 < 0:
            raise ValueError("O expoente não pode ser negativo.")
        return num1 ** num2