import pytest
from exemplo import Pessoa  # noqa: F401

# Teste para método __init__
def test_construtor():
    """
    Testa a criação de uma instância da classe Pessoa.

    Casos:
        - Nome e idade válidos.
        - Nome inválido (não string).
        - Idade inválida (negativa ou não inteira).
    """
    pessoa_valida = Pessoa("João", 25)
    assert isinstance(pessoa_valida, Pessoa)

    with pytest.raises(TypeError):
        Pessoa(123, 25)  # Nome não é uma string

    with pytest.raises(ValueError):
        Pessoa("João", -1)  # Idade é negativa
    with pytest.raises(ValueError):
        Pessoa("João", 3.5)  # Idade não é um número inteiro


# Teste para método calcular_idade
def test_calcular_idade():
    """
    Testa o cálculo da idade em anos, meses e dias.

    Casos:
        - Idades válidas.
        - Idades de borda (0 e 365).
    """
    pessoa1 = Pessoa("João", 25)
    anos, meses, dias = pessoa1.calcular_idade()
    assert anos == 25
    assert meses == 0
    assert dias == 0

    pessoa2 = Pessoa("Maria", 0)
    anos, meses, dias = pessoa2.calcular_idade()
    assert anos == 0
    assert meses == 0
    assert dias == 0

    pessoa3 = Pessoa("João", 365)
    anos, meses, dias = pessoa3.calcular_idade()
    assert anos == 1
    assert meses == 1
    assert dias == 1


# Teste para tratamento de erros no método __init__
def test_tratamento_erro():
    """
    Testa a criação de uma instância da classe Pessoa com parâmetros inválidos.

    Casos:
        - Nome e idade válidos.
        - Nome inválido (não string).
        - Idade inválida (negativa ou não inteira).
    """
    pessoa_valida = Pessoa("João", 25)
    assert isinstance(pessoa_valida, Pessoa)

    with pytest.raises(TypeError):
        Pessoa(123, 25)  # Nome não é uma string

    with pytest.raises(ValueError):
        Pessoa("João", -1)  # Idade é negativa
    with pytest.raises(ValueError):
        Pessoa("João", 3.5)  # Idade não é um número inteiro