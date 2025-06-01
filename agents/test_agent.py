from typing import Dict, Any, Optional, List, Tuple
import re
import os
import shutil
from pathlib import Path
from agents.base_agent import BaseAgent

class TestAgent(BaseAgent):
    """Agente especializado em gerar e executar testes"""
    
    def __init__(self):
        # System prompt para o agente de testes
        system_prompt = """Você é um assistente especializado em testes de software. 
Sua função é analisar código, gerar testes automatizados de alta qualidade, e executar testes.
Você deve seguir as melhores práticas de teste como:
1. Cobertura adequada (unitários, integração, e2e)
2. Nomenclatura clara e descritiva
3. Uso adequado de frameworks (pytest, unittest, jest, etc.)
4. Testes FIRST (Fast, Isolated, Repeatable, Self-verifying, Timely)
5. Uso adequado de fixtures e mocks
6. Testes negativos para verificar tratamento de erros

Responda sempre em português, com código claro e bem documentado."""
        
        super().__init__("TestAgent", system_prompt)
        
        # Mapeamento de extensões para frameworks de teste
        self.test_frameworks = {
            'py': {
                'frameworks': ['pytest', 'unittest'],
                'default': 'pytest',
                'file_pattern': 'test_{filename}.py',
                'dir_pattern': 'tests',
                'imports': {
                    'pytest': 'import pytest',
                    'unittest': 'import unittest'
                }
            },
            'js': {
                'frameworks': ['jest', 'mocha'],
                'default': 'jest',
                'file_pattern': '{filename}.test.js',
                'dir_pattern': '__tests__',
                'imports': {
                    'jest': 'const { test, expect } = require("@jest/globals");',
                    'mocha': 'const assert = require("assert");'
                }
            },
            'ts': {
                'frameworks': ['jest', 'mocha'],
                'default': 'jest',
                'file_pattern': '{filename}.test.ts',
                'dir_pattern': '__tests__',
                'imports': {
                    'jest': 'import { test, expect } from "@jest/globals";',
                    'mocha': 'import * as assert from "assert";'
                }
            },
            'java': {
                'frameworks': ['junit', 'testng'],
                'default': 'junit',
                'file_pattern': '{filename}Test.java',
                'dir_pattern': 'src/test/java',
                'imports': {
                    'junit': 'import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.*;',
                    'testng': 'import org.testng.annotations.Test;\nimport static org.testng.Assert.*;'
                }
            }
        }
        
    def can_handle(self, request: str) -> bool:
        """Verifica se este agente pode lidar com a solicitação"""
        request_lower = request.lower()
        
        # Padrões para comandos relacionados a testes
        test_patterns = [
            # Comandos explícitos de geração de testes
            r'gerar\s+testes\s+para',
            r'criar\s+testes\s+para',
            r'implementar\s+testes',
            r'escrever\s+testes',
            r'adicionar\s+testes',
            
            # Comandos de análise para testes
            r'analisar\s+.*\s+para\s+testes',
            r'verificar\s+cobertura\s+de\s+testes',
            r'testar\s+arquivo',
            r'testar\s+classe',
            r'testar\s+função',
            r'testar\s+método',
            
            # Menções a frameworks de teste
            r'\bpytest\b',
            r'\bunittest\b',
            r'\bjest\b',
            r'\bmocha\b',
            r'\bjunit\b',
            r'\btestng\b',
            
            # Termos gerais relacionados a testes
            r'\btestes\s+unitários\b',
            r'\btestes\s+de\s+integração\b',
            r'\btestes\s+e2e\b',
            r'\btestes\s+automatizados\b',
            r'\bTDD\b'
        ]
        
        # Verificar se a solicitação corresponde a algum padrão de teste
        for pattern in test_patterns:
            if re.search(pattern, request_lower):
                return True
                
        return False
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa a solicitação de testes"""
        request_lower = request.lower()
        
        # Identificar o tipo de operação
        if any(pattern in request_lower for pattern in ['gerar', 'criar', 'implementar', 'escrever', 'adicionar']):
            return self._generate_tests(request)
        elif any(pattern in request_lower for pattern in ['analisar', 'verificar', 'cobertura']):
            return self._analyze_test_coverage(request)
        elif 'executar' in request_lower or 'rodar' in request_lower:
            return self._run_tests(request)
        else:
            return self._suggest_test_strategy(request)
    
    def _generate_tests(self, request: str) -> Dict[str, Any]:
        """Gera testes para um arquivo ou código específico"""
        # Extrair o nome do arquivo para teste
        file_match = re.search(r'para\s+(?:o\s+)?(?:arquivo\s+)?([a-zA-Z0-9_\./]+\.[a-zA-Z0-9]+)', request.lower())
        
        if not file_match:
            return {
                "success": False,
                "output": "Não foi possível identificar o arquivo para o qual gerar testes. Por favor, especifique o arquivo.",
                "type": "test_generation"
            }
        
        file_path = file_match.group(1)
        
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            return {
                "success": False,
                "output": f"O arquivo '{file_path}' não foi encontrado. Verifique o caminho e tente novamente.",
                "type": "test_generation"
            }
        
        # Determinar o tipo de arquivo e framework de teste apropriado
        file_extension = os.path.splitext(file_path)[1][1:].lower()
        if file_extension not in self.test_frameworks:
            return {
                "success": False,
                "output": f"Não há suporte para testes de arquivos com extensão '{file_extension}'. As extensões suportadas são: {', '.join(self.test_frameworks.keys())}",
                "type": "test_generation"
            }
        
        # Ler o conteúdo do arquivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao ler o arquivo '{file_path}': {str(e)}",
                "type": "test_generation"
            }
        
        # Gerar nome e caminho para o arquivo de teste
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        test_framework_info = self.test_frameworks[file_extension]
        test_filename = test_framework_info['file_pattern'].format(filename=filename_without_ext)
        
        # Determinar o diretório para o arquivo de teste
        file_dir = os.path.dirname(file_path)
        if file_dir == '':
            file_dir = '.'
        
        test_dir = os.path.join(file_dir, test_framework_info['dir_pattern'])
        
        # Verificar se o diretório de testes existe, senão criar
        os.makedirs(test_dir, exist_ok=True)
        
        test_file_path = os.path.join(test_dir, test_filename)
        
        # Verificar se já existe um arquivo de teste
        test_file_exists = os.path.exists(test_file_path)
        mode = "atualizar" if test_file_exists else "criar"
        existing_test_content = ""
        
        if test_file_exists:
            try:
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    existing_test_content = f.read()
            except Exception:
                existing_test_content = ""
        
        # Framework padrão para a linguagem
        framework = test_framework_info['default']
        
        # Extrair framework explícito da solicitação, se houver
        for fw in test_framework_info['frameworks']:
            if fw.lower() in request.lower():
                framework = fw
                break
        
        # Gerar o conteúdo do teste usando LLM
        framework_import = test_framework_info['imports'][framework]
        
        # Determinar o tipo de teste a ser gerado
        test_type = "unitário"
        if "integração" in request.lower():
            test_type = "integração"
        elif "e2e" in request.lower() or "end-to-end" in request.lower():
            test_type = "end-to-end"
        
        # Criar prompt para o LLM
        prompt = f"""Gere testes {test_type}s para o seguinte código utilizando o framework {framework}.
O código está em um arquivo chamado '{filename}'. 

CÓDIGO A SER TESTADO:
```{file_extension}
{file_content}
```

{f"TESTES EXISTENTES (para referência e possível atualização):\n```{file_extension}\n{existing_test_content}\n```" if test_file_exists else ""}

INSTRUÇÕES ESPECÍFICAS:
- Utilize o framework {framework} para implementar os testes
- Inclua {framework_import} no início do arquivo
- Crie testes para todas as principais funcionalidades do código
- Inclua testes de casos normais e de borda
- Siga boas práticas como: testes isolados, nomes descritivos, assertions claras
- Use mocks ou fixtures quando apropriado
- Verifique tratamento de erros e exceções

Responda APENAS com o código dos testes, sem explicações adicionais.
"""
        
        try:
            test_code = self.invoke_llm(prompt)
            
            # Remover marcações de código, se houver
            test_code = re.sub(r'^```[a-z]*\n', '', test_code)
            test_code = re.sub(r'\n```$', '', test_code)
            
            # Salvar o arquivo de teste
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
                
            return {
                "success": True,
                "output": f"Testes {mode}dos com sucesso em '{test_file_path}' usando o framework {framework}.",
                "type": "test_generation",
                "file": test_file_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao gerar testes: {str(e)}",
                "type": "test_generation",
                "error": str(e)
            }
    
    def _analyze_test_coverage(self, request: str) -> Dict[str, Any]:
        """Analisa a cobertura de testes para um arquivo ou projeto"""
        # Extrair o nome do arquivo para análise
        file_match = re.search(r'(?:arquivo|código)\s+([a-zA-Z0-9_\./]+\.[a-zA-Z0-9]+)', request.lower())
        
        if not file_match:
            return {
                "success": False,
                "output": "Não foi possível identificar o arquivo para análise de testes. Por favor, especifique o arquivo.",
                "type": "test_analysis"
            }
        
        file_path = file_match.group(1)
        
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            return {
                "success": False,
                "output": f"O arquivo '{file_path}' não foi encontrado. Verifique o caminho e tente novamente.",
                "type": "test_analysis"
            }
        
        # Determinar o tipo de arquivo e procurar por testes associados
        file_extension = os.path.splitext(file_path)[1][1:].lower()
        if file_extension not in self.test_frameworks:
            return {
                "success": False,
                "output": f"Não há suporte para análise de testes de arquivos com extensão '{file_extension}'. As extensões suportadas são: {', '.join(self.test_frameworks.keys())}",
                "type": "test_analysis"
            }
        
        # Ler o conteúdo do arquivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao ler o arquivo '{file_path}': {str(e)}",
                "type": "test_analysis"
            }
        
        # Procurar arquivos de teste associados
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        file_dir = os.path.dirname(file_path)
        if file_dir == '':
            file_dir = '.'
        
        test_framework_info = self.test_frameworks[file_extension]
        test_dir = os.path.join(file_dir, test_framework_info['dir_pattern'])
        test_filename = test_framework_info['file_pattern'].format(filename=filename_without_ext)
        test_file_path = os.path.join(test_dir, test_filename)
        
        test_file_exists = os.path.exists(test_file_path)
        test_content = ""
        
        if test_file_exists:
            try:
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
            except Exception as e:
                return {
                    "success": False,
                    "output": f"Erro ao ler o arquivo de testes '{test_file_path}': {str(e)}",
                    "type": "test_analysis"
                }
        
        # Analisar o código e os testes usando o LLM
        prompt = f"""Analise o seguinte código e seus testes associados (se existirem) para avaliar a cobertura de testes.

CÓDIGO FONTE:
```{file_extension}
{file_content}
```

{"TESTES EXISTENTES:" if test_file_exists else "NÃO HÁ TESTES EXISTENTES PARA ESTE ARQUIVO."}
{f"```{file_extension}\n{test_content}\n```" if test_file_exists else ""}

Por favor, forneça uma análise detalhada incluindo:
1. Funcionalidades e métodos presentes no código fonte
2. Quais funcionalidades estão cobertas por testes
3. Quais funcionalidades não possuem testes (lacunas na cobertura)
4. Tipos de testes presentes (unitários, integração, etc.)
5. Qualidade dos testes existentes (assertions, mocks, fixtures, etc.)
6. Recomendações específicas para melhorar a cobertura e qualidade dos testes

Estruture sua resposta em seções claras e forneça recomendações específicas.
"""
        
        try:
            analysis = self.invoke_llm(prompt)
            
            return {
                "success": True,
                "output": analysis,
                "type": "test_analysis",
                "source_file": file_path,
                "test_file": test_file_path if test_file_exists else None,
                "has_tests": test_file_exists
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao analisar cobertura de testes: {str(e)}",
                "type": "test_analysis",
                "error": str(e)
            }
    
    def _run_tests(self, request: str) -> Dict[str, Any]:
        """Executa testes para um arquivo específico ou para todo o projeto"""
        # Implementação básica, na prática precisaria integrar com frameworks de teste específicos
        return {
            "success": False,
            "output": "A execução de testes requer integração com frameworks específicos e não está disponível nesta versão do TestAgent.",
            "type": "test_execution"
        }
    
    def _suggest_test_strategy(self, request: str) -> Dict[str, Any]:
        """Sugere uma estratégia de testes para um projeto ou componente"""
        # Extrair informações sobre o contexto/projeto da solicitação
        file_match = re.search(r'(?:arquivo|código)\s+([a-zA-Z0-9_\./]+\.[a-zA-Z0-9]+)', request.lower())
        project_type_match = re.search(r'(?:projeto|aplicação)\s+(web|api|cli|biblioteca|frontend|backend)', request.lower())
        
        file_path = file_match.group(1) if file_match else None
        project_type = project_type_match.group(1) if project_type_match else "genérico"
        
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception:
                file_content = ""
        
        # Criar prompt para o LLM
        prompt = f"""Sugira uma estratégia de testes apropriada para um projeto do tipo {project_type}.
{f"O projeto inclui o seguinte código como exemplo:\n```\n{file_content}\n```" if file_content else ""}

Inclua na sua estratégia:
1. Tipos de testes recomendados (unitários, integração, e2e, etc.)
2. Frameworks e ferramentas sugeridos
3. Estrutura de diretórios e organização dos testes
4. Práticas recomendadas específicas para este tipo de projeto
5. Estratégia para mocks, stubs e fixtures
6. Considerações sobre testes automatizados vs. manuais
7. Estratégia de integração contínua (CI) para os testes

Forneça exemplos concretos e adaptados ao contexto do projeto sempre que possível.
"""
        
        try:
            strategy = self.invoke_llm(prompt)
            
            return {
                "success": True,
                "output": strategy,
                "type": "test_strategy",
                "project_type": project_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao gerar estratégia de testes: {str(e)}",
                "type": "test_strategy",
                "error": str(e)
            }
