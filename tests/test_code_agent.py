import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ajusta o path para importar do diretório pai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.code_agent import CodeAgent

class TestCodeAgentFileOperations(unittest.TestCase):
    """Testa as operações de arquivo do CodeAgent"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.agent = CodeAgent()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
    
    def tearDown(self):
        """Limpeza após os testes"""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    def test_create_file_with_content(self):
        """Testa a criação de um arquivo com conteúdo"""
        result = self.agent._create_file("criar arquivo teste.txt com conteúdo de teste")
        self.assertTrue(result["success"])
        self.assertIn("teste.txt", result["filename"])
        
        # Verifica se o arquivo foi criado
        self.assertTrue(os.path.exists("teste.txt"))
        
        # Verifica o conteúdo do arquivo
        with open("teste.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("conteúdo de teste", content.lower())
    
    @patch('agents.code_agent.CodeAgent.invoke_llm')
    def test_edit_existing_file(self, mock_invoke_llm):
        """Testa a edição de um arquivo existente"""
        # Configura o mock para retornar um conteúdo específico
        mock_invoke_llm.return_value = "conteúdo original\nnovo conteúdo"
        
        # Cria um arquivo para editar
        test_file = "editar.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("conteúdo original")
        
        # Edita o arquivo
        result = self.agent._edit_file(f"editar o arquivo {test_file} para ter novo conteúdo")
        self.assertTrue(result["success"])
        
        # Verifica se o conteúdo foi atualizado
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verifica se o conteúdo contém o esperado
        self.assertIn("novo conteúdo", content.lower())
        self.assertIn("conteúdo original", content.lower())
        
        # Limpa o arquivo de teste
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_read_file(self):
        """Testa a leitura de um arquivo existente"""
        # Cria um arquivo para ler
        with open("ler.txt", 'w', encoding='utf-8') as f:
            f.write("conteúdo para leitura")
        
        # Lê o arquivo
        result = self.agent._read_file("ler o arquivo ler.txt")
        self.assertTrue(result["success"])
        self.assertIn("conteúdo para leitura", result["output"])

class TestCodeAgentTestOperations(unittest.TestCase):
    """Testa as operações de teste do CodeAgent"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.agent = CodeAgent()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Cria uma estrutura de teste simples
        os.makedirs("tests", exist_ok=True)
        with open("tests/test_example.py", 'w', encoding='utf-8') as f:
            f.write("""
import unittest

def test_example():
    assert 1 + 1 == 2

class TestExample(unittest.TestCase):
    def test_example_class(self):
        self.assertEqual(2 + 2, 4)
""")
    
    def tearDown(self):
        """Limpeza após os testes"""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    @patch('subprocess.run')
    def test_run_tests(self, mock_run):
        """Testa a execução de testes"""
        # Cria um diretório temporário para o teste
        test_dir = "test_dir"
        os.makedirs(test_dir, exist_ok=True)
        
        # Cria um arquivo de teste temporário
        test_file = os.path.join(test_dir, "test_example.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("def test_example():\n    assert True")
        
        # Configura o mock para simular a execução do pytest
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{test_file}::test_example PASSED [100%]\n\n1 passed in 0.01s",
            stderr=""
        )
        
        # Executa os testes
        result = self.agent._handle_test_request(f"executar testes em {test_file}")
        
        # Verifica o resultado
        self.assertTrue(result["success"], f"Teste falhou com saída: {result.get('output', 'Nenhuma saída')}")
        
        # Verifica se o nome do arquivo de teste está na saída
        self.assertIn("test_example.py", result["output"], 
                     f"Nome do arquivo de teste não encontrado na saída: {result['output']}")
        
        # Verifica se o teste passou
        self.assertTrue(result["passed"], "O teste deveria ter passado")
        
        # Verifica se o mock foi chamado corretamente
        mock_run.assert_called_once()
        
        # Limpa os arquivos de teste
        os.remove(test_file)
        os.rmdir(test_dir)

class TestCodeAgentProjectOperations(unittest.TestCase):
    """Testa as operações de projeto do CodeAgent"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.agent = CodeAgent()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
    
    def tearDown(self):
        """Limpeza após os testes"""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    def test_create_python_project_structure(self):
        """Testa a criação de uma estrutura de projeto Python"""
        result = self.agent._create_project_structure("criar projeto python")
        
        # Verifica se a estrutura foi criada com sucesso
        self.assertTrue(result["success"])
        self.assertEqual(result["project_type"], "python")
        
        # Verifica se os arquivos e diretórios foram criados
        expected_items = [
            "README.md",
            "requirements.txt",
            "setup.py",
            "src/",
            "tests/"
        ]
        
        for item in expected_items:
            if item.endswith('/'):
                self.assertTrue(os.path.isdir(item[:-1]))
            else:
                self.assertTrue(os.path.isfile(item))
    
    def test_show_project_structure(self):
        """Testa a exibição da estrutura do projeto"""
        # Cria alguns arquivos e diretórios
        os.makedirs("src", exist_ok=True)
        os.makedirs("tests", exist_ok=True)
        with open("main.py", 'w') as f:
            f.write("")
        
        # Obtém a estrutura
        result = self.agent._show_project_structure()
        
        # Verifica o resultado
        self.assertTrue(result["success"])
        self.assertIn("src", result["output"])
        self.assertIn("tests", result["output"])
        self.assertIn("main.py", result["output"])

class TestCodeAgentCodeAnalysis(unittest.TestCase):
    """Testa as operações de análise de código do CodeAgent"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.agent = CodeAgent()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Cria um arquivo Python para análise
        with open("example.py", 'w', encoding='utf-8') as f:
            f.write("""
def calculate_sum(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
""")
    
    def tearDown(self):
        """Limpeza após os testes"""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    @patch('agents.code_agent.CodeAgent.invoke_llm')
    def test_analyze_code(self, mock_invoke_llm):
        """Testa a análise de um arquivo de código"""
        # Configura o mock para retornar uma análise simulada
        mock_invoke_llm.return_value = """
        Resumo: O código contém uma função e uma classe simples.
        Problemas: Nenhum problema encontrado.
        Sugestões: Adicionar documentação e testes.
        """
        
        # Executa a análise
        result = self.agent._analyze_code("analisar example.py")
        
        # Verifica o resultado
        self.assertTrue(result["success"])
        self.assertEqual(result["filename"], "example.py")
        self.assertIn("Resumo", result["output"])
        
        # Verifica se o método invoke_llm foi chamado com o conteúdo correto
        mock_invoke_llm.assert_called_once()
        call_args = mock_invoke_llm.call_args[0][0]
        self.assertIn("def calculate_sum", call_args)
        self.assertIn("class Calculator", call_args)
    
    @patch('agents.code_agent.CodeAgent.invoke_llm')
    def test_refactor_code(self, mock_invoke_llm):
        """Testa a refatoração de código"""
        # Configura o mock para retornar um código refatorado
        mock_invoke_llm.return_value = '''
def calculate_sum(a: int, b: int) -> int:
    """Soma dois números inteiros.
    
    Args:
        a: Primeiro número
        b: Segundo número
        
    Returns:
        A soma dos dois números
    """
    return a + b

class Calculator:
    """Calculadora simples com operações básicas."""
    
    def multiply(self, x: float, y: float) -> float:
        """Multiplica dois números.
        
        Args:
            x: Primeiro fator
            y: Segundo fator
            
        Returns:
            O produto dos dois números
        """
        return x * y
'''
        
        # Executa a refatoração
        result = self.agent._refactor_code("refatorar example.py adicionando type hints e docstrings")
        
        # Verifica o resultado
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists("example.py.bak"))  # Verifica o backup
        
        # Verifica se o arquivo foi atualizado
        with open("example.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("def calculate_sum(a: int, b: int) -> int", content)
        self.assertIn('Soma dois números inteiros', content)
        self.assertIn('class Calculator:', content)
        
        # Verifica se o diff foi gerado
        self.assertIn("diff", result)
        self.assertIn("original/example.py", result["diff"])
        self.assertIn("refactored/example.py", result["diff"])

class TestCodeAgentIntegration(unittest.TestCase):
    """Testes de integração para fluxos completos do CodeAgent"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.agent = CodeAgent()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Configura mocks para evitar chamadas reais ao LLM
        self.llm_patcher = patch('agents.code_agent.CodeAgent.invoke_llm')
        self.mock_invoke_llm = self.llm_patcher.start()
        
        # Configura mocks para comandos do sistema
        self.subprocess_patcher = patch('subprocess.run')
        self.mock_subprocess = self.subprocess_patcher.start()
    
    def tearDown(self):
        """Limpeza após os testes"""
        self.llm_patcher.stop()
        self.subprocess_patcher.stop()
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    def test_end_to_end_workflow(self):
        """Testa um fluxo completo: criar, editar, testar e refatorar código"""
        # Configura o mock do LLM para retornar diferentes valores baseado na entrada
        def mock_llm_side_effect(prompt, **kwargs):
            if "criar arquivo main.py" in prompt:
                return """
                # main.py
                def hello():
                    return "Hello, World!"
                    
                if __name__ == "__main__":
                    print(hello())
                """
            elif "criar arquivo test_main.py" in prompt:
                return """
                import unittest
                from main import hello
                
                class TestHello(unittest.TestCase):
                    def test_hello(self):
                        self.assertEqual(hello(), "Hello, World!")
                        
                if __name__ == "__main__":
                    unittest.main()
                """
            elif "analisar código" in prompt:
                return """
                Análise de código concluída. Aqui estão as recomendações:
                1. A função hello poderia ter uma docstring
                2. Considere adicionar type hints
                """
            elif "refatorar" in prompt:
                return """
                def hello() -> str:
                    \"\"\"Retorna uma saudação em inglês.\"\"\"
                    return "Hello, World!"
                    
                if __name__ == "__main__":
                    print(hello())
                """
            return ""
        
        self.mock_invoke_llm.side_effect = mock_llm_side_effect
        
        # 1. Cria o arquivo principal
        print("\n=== Test: Criando main.py ===")
        result = self.agent._create_file("criar arquivo main.py com uma função hello que retorna 'Hello, World!'")
        print(f"Resultado da criação do arquivo: {result}")
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists("main.py"))
        
        # 2. Cria o arquivo de teste
        print("\n=== Test: Criando test_main.py ===")
        result = self.agent._create_file("criar arquivo test_main.py com testes para a função hello")
        print(f"Resultado da criação do teste: {result}")
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists("test_main.py"))
        
        # 3. Execução de testes (mockando o subprocess para simular sucesso)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Ran 1 test in 0.000s\nOK"
        self.mock_subprocess.return_value = mock_result
        
        # Executa os testes diretamente
        result = self.agent._handle_test_request("executar testes em test_main.py")
        print(f"Resultado da execução dos testes: {result}")
        self.assertTrue(result["success"])
        self.assertIn("1 test", result["output"])
        
        # 4. Análise de código
        result = self.agent._analyze_code("analisar código em main.py")
        print(f"Resultado da análise de código: {result}")
        self.assertTrue(result["success"])
        self.assertIn("análise", result["output"].lower())
        
        # 5. Refatoração de código
        result = self.agent._refactor_code("refatorar main.py para adicionar type hints e docstrings")
        print(f"Resultado da refatoração: {result}")
        self.assertTrue(result["success"])
        self.assertIn("refatorado", result["output"].lower())
        
        # Verifica se o arquivo foi realmente modificado
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('"""', content)  # Verifica se tem docstring
            self.assertIn('-> str', content)  # Verifica se tem type hint
    
    def test_file_operations_workflow(self):
        """Testa um fluxo completo de operações de arquivo"""
        # Configura o mock para retornar conteúdo de teste
        self.mock_invoke_llm.return_value = "conteúdo inicial"
        
        # 1. Cria um arquivo
        print("\n=== Test: Criando teste.txt ===")
        # Usamos _create_file diretamente para evitar a execução de testes
        result = self.agent._create_file("criar arquivo teste.txt com conteúdo inicial")
        print(f"Resultado da criação do arquivo: {result}")
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists("teste.txt"))
        
        # 2. Lê o arquivo
        result = self.agent._read_file("ler conteúdo de teste.txt")
        self.assertTrue(result["success"])
        self.assertIn("conteúdo inicial", result["output"].lower())
        
        # 3. Edita o arquivo
        self.mock_invoke_llm.return_value = "conteúdo inicial\nmais conteúdo adicionado"
        # Configura o mock para simular sucesso na edição
        mock_edit_result = MagicMock()
        mock_edit_result.returncode = 0
        self.mock_subprocess.return_value = mock_edit_result
        
        # Usamos _edit_file diretamente para evitar a execução de testes
        result = self.agent._edit_file("editar teste.txt para adicionar mais conteúdo")
        self.assertTrue(result["success"])
        
        # 4. Verifica se o arquivo foi editado corretamente
        with open("teste.txt", 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("mais conteúdo adicionado", content.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
