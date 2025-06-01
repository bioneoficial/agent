#!/usr/bin/env python3
"""
Classe representando uma pessoa com nome e idade.
"""

class Pessoa:
    """
    Classe responsável por armazenar informações sobre uma pessoa.

    Atributos:
        - nome (str): Nome da pessoa.
        - idade (int): Idade da pessoa.

    Métodos:
        - __init__: Inicializa os atributos da classe.
        - calcular_idade: Calcula a idade em anos, meses e dias.
    """

    def __init__(self, nome, idade):
        """
        Inicializa a classe com o nome e idade da pessoa.

        Parâmetros:
            - nome (str): Nome da pessoa.
            - idade (int): Idade da pessoa.

        Erros:
            - TypeError: Se algum parâmetro não for do tipo esperado.
            - ValueError: Se a idade não for um número inteiro positivo.
        """
        if not isinstance(nome, str):
            raise TypeError("Nome precisa ser uma string")
        if not isinstance(idade, (int, float)):
            raise TypeError("Idade precisa ser um número")
        if not isinstance(idade, int) or idade < 0:
            raise ValueError("Idade precisa ser um número inteiro positivo")

        self.nome = nome
        self.idade = idade


    def calcular_idade(self):
        """
        Calcula a idade da pessoa em anos, meses e dias.

        Retorno:
            - Tupla com os valores de idade em anos, meses e dias.
        """

        anos = self.idade // 365
        meses = (self.idade % 365) // 30
        dias = (self.idade % 365) % 30

        return anos, meses, dias


# Teste da classe Pessoa
pessoa1 = Pessoa("João", 25)
anos, meses, dias = pessoa1.calcular_idade()
print(f"A idade de {pessoa1.nome} é: {anos} ano(s), {meses} mês(es) e {dias} dia(s)")