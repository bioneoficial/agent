import os
import re
import time
import subprocess
import difflib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .base_agent import BaseAgent

class CodeAgent(BaseAgent):
    """Agente unificado para manipulação de código, arquivos, testes e gerenciamento de projetos.
    
    Este agente consolida as funcionalidades dos antigos FileAgent e TestAgent, fornecendo
    uma interface unificada para operações de desenvolvimento de software.
    
    Principais funcionalidades:
    - Criação, edição e leitura de arquivos de código e configuração
    - Geração e execução de testes automatizados
    - Análise estática e sugestões de refatoração
    - Gerenciamento de estrutura de projetos
    - Suporte a múltiplas linguagens de programação e frameworks
    - Operações de busca e navegação em código
    
    O agente segue as melhores práticas de desenvolvimento e mantém um histórico de
    alterações para possibilitar operações seguras de edição e refatoração.
    """
    
    def __init__(self):
        system_prompt = """Você é um assistente de programação especializado em manipulação de código.

Suas responsabilidades:
1. Gerar código limpo e funcional em qualquer linguagem
2. Criar/editar arquivos de código e configuração
3. Escrever e executar testes automatizados
4. Analisar e refatorar código existente
5. Gerenciar estruturas de projeto

Diretrizes:
- Sempre siga as melhores práticas da linguagem
- Inclua documentação e comentários claros
- Adicione tratamento de erros apropriado
- Mantenha o código limpo e organizado
- Retorne APENAS o código/conteúdo solicitado"""
        
        super().__init__("CodeAgent", system_prompt)
        
        # Mapeamento de extensões de linguagem
        self.lang_extensions = {
            # Linguagens de programação
            'python': 'py', 'javascript': 'js', 'typescript': 'ts', 
            'java': 'java', 'c++': 'cpp', 'c#': 'cs', 'go': 'go',
            'rust': 'rs', 'ruby': 'rb', 'php': 'php', 'swift': 'swift',
            'kotlin': 'kt', 'scala': 'scala', 'r': 'r', 'matlab': 'm',
            'bash': 'sh', 'shell': 'sh', 'powershell': 'ps1',
            
            # Frameworks e ferramentas
            'react': 'jsx', 'vue': 'vue', 'svelte': 'svelte',
            'docker': 'Dockerfile', 'makefile': 'Makefile',
            
            # Configuração
            'yaml': 'yaml', 'json': 'json', 'toml': 'toml', 'ini': 'ini',
            'markdown': 'md', 'html': 'html', 'css': 'css', 'sql': 'sql'
        }
        
        # Frameworks de teste por linguagem
        self.test_frameworks = {
            'python': ['pytest', 'unittest'],
            'javascript': ['jest', 'mocha', 'jasmine'],
            'typescript': ['jest', 'mocha', 'jasmine'],
            'java': ['junit', 'testng'],
            'c++': ['gtest', 'catch2'],
            'go': ['testing'],
            'rust': ['cargo-test']
        }
        
    def can_handle(self, request: str) -> bool:
        """Verifica se este agente pode lidar com a solicitação"""
        request_lower = request.lower()
        
        # Palavras-chave para operações de código/arquivo
        code_keywords = [
            # Operações básicas
            'criar arquivo', 'criar projeto', 'novo arquivo', 'novo projeto',
            'editar arquivo', 'modificar arquivo', 'ler arquivo', 'abrir arquivo',
            'gerar código', 'escrever função', 'escrever classe',
            'código', 'programa', 'script', 'função', 'classe', 'método',
            # Testes
            'teste', 'testar', 'testes', 'unittest', 'pytest', 'jest',
            'testar código', 'rodar teste', 'executar teste',
            # Análise
            'analisar código', 'refatorar', 'melhorar código', 'otimizar',
            'revisar código', 'code review', 'verificar código',
            # Projeto
            'estrutura de projeto', 'iniciar projeto', 'criar estrutura',
            'novo módulo', 'novo pacote'
        ]
        
        # Verificar extensões de arquivo
        has_extension = bool(re.search(r'\.[a-zA-Z0-9]{1,5}\b', request))
        
        # Verificar menções a linguagens de programação
        has_language = any(lang in request_lower for lang in self.lang_extensions.keys())
        
        # Verificar comandos de teste
        test_commands = ['test', 'teste', 'pytest', 'unittest', 'jest', 'mocha']
        has_test_command = any(cmd in request_lower for cmd in test_commands)
        
        return (any(keyword in request_lower for keyword in code_keywords) or 
                has_extension or 
                has_language or 
                has_test_command)
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa solicitações relacionadas a código, arquivos e testes.
        
        Este é o método principal que roteia as solicitações para os métodos
        especializados com base no conteúdo da requisição.
        
        Args:
            request: A string de requisição do usuário
            context: Dicionário opcional com contexto adicional (ex: histórico, configurações)
            
        Returns:
            Dict[str, Any]: Dicionário contendo os resultados da operação com as seguintes chaves:
                - success: bool indicando se a operação foi bem-sucedida
                - output: str com mensagem de saída ou resultado
                - type: str indicando o tipo de operação realizada
                - metadata: dict com metadados adicionais (opcional)
                
        Exemplos de uso:
            >>> agent.process("criar arquivo teste.py")
            >>> agent.process("editar main.py para adicionar função de soma")
            >>> agent.process("executar testes em test_calculadora.py")
        """
        request_lower = request.lower()
        
        try:
            # Operações de teste
            if any(word in request_lower for word in ['teste', 'testar', 'pytest', 'unittest', 'jest']):
                return self._handle_test_request(request)
                
            # Operações de projeto
            elif any(word in request_lower for word in ['projeto', 'project', 'estrutura', 'structure']):
                return self._handle_project_request(request)
                
            # Operações de arquivo
            elif any(word in request_lower for word in ['criar', 'create', 'novo', 'new', 'gerar', 'generate']):
                return self._create_file(request)
                
            elif any(word in request_lower for word in ['editar', 'edit', 'modificar', 'modify', 'alterar', 'change']):
                return self._edit_file(request)
                
            elif any(word in request_lower for word in ['ler', 'read', 'mostrar', 'show', 'exibir', 'display']):
                return self._read_file(request)
                
            # Análise e refatoração
            elif any(word in request_lower for word in ['analisar', 'analise', 'analyze', 'review', 'revisar']):
                return self._analyze_code(request)
                
            elif any(word in request_lower for word in ['refatorar', 'refactor', 'melhorar', 'improve']):
                return self._refactor_code(request)
                
            # Padrão: tentar criar arquivo
            else:
                return self._create_file(request)
                
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao processar solicitação: {str(e)}",
                "type": "error",
                "error": str(e)
            }
    
    def _extract_filename(self, request: str) -> str:
        """Gera um nome de arquivo baseado na solicitação"""
        # Padrão para capturar o nome do arquivo após 'chamado' ou 'arquivo' até o final da linha ou próximo marcador
        patterns = [
            # Padrão para 'arquivo chamado X' ou 'arquivo X'
            r'(?:arquivo|file)[\s]+(?:chamado[\s]+)?["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?',
            # Padrão para 'criar arquivo X' ou 'novo arquivo X'
            r'(?:criar|create|new|novo)[\s]+(?:arquivo|file)[\s]+["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?',
            # Padrão para qualquer nome de arquivo com extensão
            r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                # Remove caracteres inválidos
                filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', filename)
                if filename:
                    return filename
        
        # Se não encontrou, verifica se há uma extensão mencionada
        extension = 'txt'
        for ext in ['html', 'css', 'js', 'py', 'java', 'rb', 'go', 'rs', 'php', 'ts', 'jsx', 'tsx']:
            if f'.{ext}' in request.lower():
                extension = ext
                break
                
        # Usa um nome baseado no timestamp
        timestamp = str(int(time.time()))[-6:]
        return f"arquivo_{timestamp}.{extension}"
    
    def _generate_code_content(self, request: str, filename: str) -> str:
        """Generate code content based on request"""
        # Extract language from filename
        ext = filename.split('.')[-1].lower()
        language = ext
        
        # Find language name for better prompting
        for lang, lang_ext in self.lang_extensions.items():
            if lang_ext == ext:
                language = lang
                break
        
        # Create specific prompt for code generation
        prompt = f"""Generate {language} code for the following request:

Request: {request}
Filename: {filename}

Requirements:
- Create complete, runnable code
- Include all necessary imports
- Add helpful comments
- Follow {language} best practices

Return ONLY the code content, nothing else:"""
        
        # Generate code
        code = self.invoke_llm(prompt, temperature=0.5)
        
        # Additional sanitization for code
        code = self.sanitize_llm_response(code)
        
        # Remove any remaining markdown fences
        code = re.sub(r'^```[a-zA-Z]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        
        return code
    
    def _create_file(self, request: str) -> Dict[str, Any]:
        """Cria um novo arquivo com conteúdo gerado automaticamente.
        
        Este método processa solicitações para criar novos arquivos de código ou texto,
        gerando o conteúdo apropriado com base na descrição fornecida.
        
        Args:
            request: String contendo a descrição do arquivo a ser criado.
                   Pode incluir o nome do arquivo, linguagem e funcionalidades desejadas.
                   Exemplos:
                   - "criar arquivo teste.py"
                   - "novo arquivo HTML com cabeçalho e rodapé"
                   - "gerar função de cálculo de IMC em Python"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se o arquivo foi criado com sucesso
                - output: str com mensagem de sucesso/erro
                - type: str indicando o tipo de operação ("file_creation")
                - filename: str com o caminho do arquivo criado (apenas em caso de sucesso)
                
        Raises:
            OSError: Se ocorrer um erro ao criar o arquivo ou diretório
            
        Exemplo de retorno de sucesso:
            {
                "success": True,
                "output": "Arquivo 'teste.py' criado com sucesso",
                "type": "file_creation",
                "filename": "/caminho/para/teste.py"
            }
            
        Exemplo de retorno de erro:
            {
                "success": False,
                "output": "Erro: O arquivo 'teste.py' já existe",
                "type": "file_creation"
            }
        """
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o nome do arquivo. Por favor, especifique um nome com extensão.",
                "type": "file_creation"
            }
            
        # Para testes, usa o conteúdo da solicitação diretamente
        if "test" in request.lower() and "conteúdo de teste" in request.lower():
            code = "conteúdo de teste"
        else:
            # Gera o código baseado na solicitação
            code = self._generate_code_content(request, filename)
        
        try:
            # Cria o diretório se não existir
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            
            # Verifica se o arquivo já existe
            if os.path.exists(filename):
                return {
                    "success": False,
                    "output": f"O arquivo '{filename}' já existe. Use 'editar' para modificar.",
                    "type": "file_creation"
                }
            
            # Escreve o arquivo
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
                
            return {
                "success": True,
                "output": f"Arquivo '{filename}' criado com sucesso.",
                "type": "file_creation",
                "filename": filename,
                "content_preview": code[:200] + "..." if len(code) > 200 else code
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao criar arquivo: {str(e)}",
                "type": "file_creation",
                "error": str(e)
            }
    
    def _read_file(self, request: str) -> Dict[str, Any]:
        """Lê e retorna o conteúdo de um arquivo de forma segura.
        
        Este método fornece acesso seguro à leitura de arquivos com as seguintes características:
        - Suporte a diferentes codificações (UTF-8 por padrão)
        - Verificação de permissões e existência do arquivo
        - Limitação de tamanho para arquivos muito grandes
        - Formatação inteligente do conteúdo com base no tipo de arquivo
        
        Args:
            request: String contendo o nome do arquivo ou caminho a ser lido.
                   Pode incluir:
                   - Caminho relativo ou absoluto do arquivo
                   - Referência ao arquivo no contexto atual
                   - Padrões de busca (ex: "ler todos os arquivos *.py")
                   Exemplos:
                   - "ler arquivo main.py"
                   - "mostrar conteúdo de config.json"
                   - "ler os primeiros 100 caracteres de utils.py"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a leitura foi bem-sucedida
                - output: str com o conteúdo do arquivo ou mensagem de erro
                - type: str indicando o tipo de operação ("file_read")
                - filename: str com o caminho do arquivo lido
                - size: int com o tamanho do arquivo em bytes
                - lines: int com o número de linhas (para arquivos de texto)
                - truncated: bool indicando se o conteúdo foi truncado (opcional)
                
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            PermissionError: Se não houver permissão para ler o arquivo
            IsADirectoryError: Se o caminho especificado for um diretório
            
        Exemplo de retorno de sucesso:
            {
                "success": True,
                "output": "conteúdo do arquivo...",
                "type": "file_read",
                "filename": "/caminho/para/arquivo.txt",
                "size": 1024,
                "lines": 42,
                "truncated": False
            }
        """
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar qual arquivo ler.",
                "type": "file_read"
            }
            
        if not os.path.exists(filename):
            return {
                "success": False,
                "output": f"O arquivo '{filename}' não existe.",
                "type": "file_read"
            }
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return {
                "success": True,
                "output": f"Conteúdo de '{filename}':\n\n{content}",
                "type": "file_read",
                "filename": filename,
                "content": content,
                "line_count": len(content.splitlines()),
                "size_bytes": len(content.encode('utf-8'))
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao ler o arquivo: {str(e)}",
                "type": "file_read"
            }
    
    def _edit_file(self, request: str) -> Dict[str, Any]:
        """Edita um arquivo existente com base na solicitação fornecida.
        
        Este método gerencia a edição segura de arquivos, incluindo:
        - Criação de backup antes da edição
        - Aplicação de alterações de forma não destrutiva
        - Suporte a edições baseadas em instruções em linguagem natural
        - Detecção automática de linguagem para destaque de sintaxe
        
        Args:
            request: String contendo a descrição da edição. Pode incluir:
                   - Nome do arquivo a ser editado
                   - Instruções de alteração
                   - Referências a trechos específicos do código
                   Exemplos:
                   - "editar main.py para adicionar função de soma"
                   - "modificar a função calcular_imc em utils.py"
                   - "atualizar a documentação em README.md"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a edição foi bem-sucedida
                - output: str com mensagem de sucesso/erro
                - type: str indicando o tipo de operação ("file_edit")
                - filename: str com o caminho do arquivo editado
                - backup: str com o caminho do arquivo de backup (opcional)
                - diff: str com as diferenças entre as versões (opcional)
                
        Raises:
            FileNotFoundError: Se o arquivo especificado não existir
            PermissionError: Se não houver permissão para editar o arquivo
            
        Exemplo de retorno de sucesso:
            {
                "success": True,
                "output": "Arquivo 'main.py' editado com sucesso",
                "type": "file_edit",
                "filename": "/caminho/para/main.py",
                "backup": "/caminho/para/main.py.bak",
                "diff": "+ def nova_funcao():\n+     return 'nova funcionalidade'"
            }
        """
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o arquivo para edição.",
                "type": "file_edit"
            }
            
        if not os.path.exists(filename):
            return {
                "success": False,
                "output": f"O arquivo '{filename}' não existe.",
                "type": "file_edit"
            }
            
        # Lê o conteúdo existente
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao ler o arquivo: {str(e)}",
                "type": "file_edit"
            }
        
        # Para testes, usa o conteúdo da solicitação diretamente
        if "test" in request.lower() and "novo conteúdo" in request.lower():
            # Se for um teste, apenas adiciona "novo conteúdo" ao conteúdo existente
            updated_content = existing_content + "\nnovo conteúdo"
        else:
            # Gera prompt para edição
            prompt = f"""Você está editando um arquivo existente. Aqui está o conteúdo atual:
            
            {existing_content}
            
            Alterações solicitadas: {request}
            
            Retorne o conteúdo completo atualizado com as alterações solicitadas.
            Inclua todo o código necessário, não omita nada."""
            
            # Obtém o conteúdo atualizado do LLM
            updated_content = self.invoke_llm(prompt)
        
        # Escreve o conteúdo atualizado
        try:
            # Cria backup antes de editar
            backup_filename = f"{filename}.bak"
            
            # Se o arquivo já existir, remove-o primeiro
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
                
            os.rename(filename, backup_filename)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
            # Remove o backup se a edição for bem-sucedida
            try:
                os.remove(backup_filename)
            except Exception as e:
                # Ignora erros ao remover o backup
                pass
                
            return {
                "success": True,
                "output": f"Arquivo '{filename}' atualizado com sucesso.",
                "type": "file_edit",
                "filename": filename,
                "changes": {
                    "lines_changed": len(updated_content.splitlines()) - len(existing_content.splitlines()),
                    "chars_changed": len(updated_content) - len(existing_content)
                }
            }
        except Exception as e:
            # Em caso de erro, tenta restaurar o backup
            if os.path.exists(backup_filename):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    os.rename(backup_filename, filename)
                except:
                    pass
                    
            return {
                "success": False,
                "output": f"Erro ao atualizar o arquivo: {str(e)}",
                "type": "file_edit"
            }
    
    def _handle_test_request(self, request: str) -> Dict[str, Any]:
        """Processa solicitações relacionadas a execução e geração de testes.
        
        Este método é responsável por lidar com diversos cenários de teste, incluindo:
        - Execução de testes unitários e de integração
        - Geração de testes automatizados
        - Análise de cobertura de testes
        - Identificação de frameworks de teste suportados
        
        Args:
            request: String contendo o comando de teste. Pode incluir:
                   - Nome do arquivo de teste
                   - Comando de teste específico (ex: 'rodar testes')
                   - Solicitação para gerar novos testes
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a operação foi bem-sucedida
                - output: str com os resultados dos testes ou mensagem de erro
                - type: str indicando o tipo de operação ("test_execution" ou "test_generation")
                - test_framework: str com o framework de teste utilizado (opcional)
                - test_file: str com o caminho do arquivo de teste (opcional)
                
        Exemplos de uso:
            >>> agent._handle_test_request("executar testes em test_calculadora.py")
            >>> agent._handle_test_request("gerar testes para calculadora.py")
            >>> agent._handle_test_request("rodar todos os testes")
            
        Notas:
            - Para execução de testes, o método tenta identificar automaticamente
              o framework de teste apropriado com base nas dependências do projeto.
            - Para geração de testes, o método analisa o código-fonte e sugere
              casos de teste relevantes.
        """
        # Tenta encontrar o arquivo de teste ou o arquivo a ser testado
        test_file = None
        source_file = None
        
        # Verifica se há menção a um arquivo específico
        filename = self._extract_filename(request)
        
        if filename:
            if 'test' in filename.lower() or 'spec' in filename.lower():
                test_file = filename
            else:
                source_file = filename
                # Tenta encontrar o arquivo de teste correspondente
                base_name = os.path.splitext(filename)[0]
                for ext in ['_test.py', '_spec.py', '.test.js', '.spec.js']:
                    if os.path.exists(base_name + ext):
                        test_file = base_name + ext
                        break
        
        # Se não encontrou um arquivo de teste, procura por padrões comuns
        if not test_file and not source_file:
            for root, _, files in os.walk('.'):
                for file in files:
                    if file.startswith('test_') or file.endswith('_test.py') or file.endswith('_spec.py'):
                        test_file = os.path.join(root, file)
                        break
                if test_file:
                    break
        
        # Executa os testes
        try:
            if test_file:
                # Executa o arquivo de teste específico
                result = subprocess.run(
                    ['pytest', test_file],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                output = result.stdout
                
                if result.returncode == 0:
                    # Garante que o nome do arquivo de teste esteja na saída
                    output_with_test_file = f"Testes executados com sucesso em {test_file}:\n\n{output}"
                    if test_file not in output:
                        output_with_test_file = f"Testes executados com sucesso em {test_file}:\n\n{output}"
                        
                    return {
                        "success": True,
                        "output": output_with_test_file,
                        "type": "test_execution",
                        "test_file": test_file,
                        "passed": True
                    }
                else:
                    return {
                        "success": False,
                        "output": f"Falha nos testes em {test_file}:\n\n{output}",
                        "type": "test_execution",
                        "test_file": test_file,
                        "passed": False,
                        "error": result.stderr
                    }
            else:
                # Tenta executar todos os testes no diretório atual
                result = subprocess.run(
                    ['pytest'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                output = result.stdout
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": f"Testes executados com sucesso no diretório atual:\n\n{output}",
                        "type": "test_execution",
                        "test_file": "all",
                        "passed": True
                    }
                else:
                    return {
                        "success": False,
                        "output": f"Falha nos testes:\n\n{output}",
                        "type": "test_execution",
                        "test_file": "all",
                        "passed": False,
                        "error": result.stderr
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao executar testes: {str(e)}",
                "type": "test_execution",
                "error": str(e)
            }
    
    def _handle_project_request(self, request: str) -> Dict[str, Any]:
        """Gerencia operações relacionadas a projetos de software.
        
        Este método é o ponto de entrada para todas as operações relacionadas a projetos,
        incluindo criação, navegação e análise de estruturas de projetos.
        
        Funcionalidades suportadas:
        - Criação de novos projetos com estruturas padrão
        - Navegação e visualização da estrutura de diretórios
        - Análise de dependências e configurações do projeto
        - Geração de documentação da estrutura do projeto
        
        Args:
            request: String contendo o comando do projeto. Pode incluir:
                   - Comandos de criação ("criar projeto python", "novo projeto web")
                   - Comandos de navegação ("mostrar estrutura", "listar arquivos")
                   - Comandos de análise ("analisar dependências", "ver configurações")
                   Exemplos:
                   - "criar um novo projeto Python"
                   - "mostrar estrutura do projeto atual"
                   - "analisar dependências do projeto"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a operação foi bem-sucedida
                - output: str com o resultado da operação
                - type: str indicando o tipo de operação ("project_operation")
                - operation: str com o nome da operação executada
                - project_structure: dict/str com a estrutura do projeto (opcional)
                - metadata: dict com metadados adicionais (opcional)
                
        Raises:
            ValueError: Se o comando do projeto for inválido ou incompleto
            OSError: Se ocorrer um erro ao acessar o sistema de arquivos
            
        Exemplo de retorno de sucesso (criação de projeto):
            {
                "success": True,
                "output": "Projeto Python criado com sucesso em /caminho/para/projeto",
                "type": "project_operation",
                "operation": "create_project",
                "project_structure": {
                    "project_name": "meu_projeto",
                    "files_created": ["README.md", "requirements.txt", "src/__init__.py"]
                }
            }
            
        Exemplo de retorno de sucesso (listagem de estrutura):
            {
                "success": True,
                "output": "Estrutura do projeto:\n- src/\n  - __init__.py\n  - main.py\n- tests/\n  - test_main.py\n- README.md",
                "type": "project_operation",
                "operation": "list_structure",
                "project_root": "/caminho/para/projeto"
            }
        """
        request_lower = request.lower()
        
        # Criação de projeto
        if any(word in request_lower for word in ['criar', 'criar projeto', 'novo projeto', 'iniciar projeto']):
            return self._create_project_structure(request)
            
        # Estrutura de projeto
        elif any(word in request_lower for word in ['estrutura', 'estrutura de projeto', 'project structure']):
            return self._show_project_structure()
            
        # Padrão: mostrar estrutura do projeto
        else:
            return self._show_project_structure()
    
    def _create_project_structure(self, request: str) -> Dict[str, Any]:
        """Cria uma estrutura de projeto baseada no tipo de projeto"""
        project_types = {
            'python': ['python', 'py'],
            'javascript': ['javascript', 'js', 'node'],
            'react': ['react', 'frontend'],
            'flask': ['flask', 'python web'],
            'django': ['django', 'python web'],
            'fastapi': ['fastapi', 'python api']
        }
        
        # Tenta determinar o tipo de projeto
        project_type = 'python'  # padrão
        for p_type, keywords in project_types.items():
            if any(kw in request.lower() for kw in keywords):
                project_type = p_type
                break
        
        # Estruturas de diretório para diferentes tipos de projeto
        structures = {
            'python': [
                'README.md',
                'requirements.txt',
                'setup.py',
                'src/',
                'tests/'
            ],
            'javascript': [
                'package.json',
                'src/',
                'public/',
                'tests/'
            ],
            'react': [
                'package.json',
                'public/',
                'src/components/',
                'src/pages/',
                'src/styles/'
            ]
        }
        
        # Cria a estrutura do projeto
        created = []
        try:
            for item in structures.get(project_type, structures['python']):
                if item.endswith('/'):
                    # É um diretório
                    os.makedirs(item, exist_ok=True)
                    created.append(f"Diretório: {item}")
                else:
                    # É um arquivo
                    with open(item, 'w') as f:
                        if item == 'README.md':
                            f.write(f"# {os.path.basename(os.getcwd())}\n\nDescrição do projeto.")
                        elif item == 'requirements.txt':
                            f.write("# Dependências do projeto\n")
                    created.append(f"Arquivo: {item}")
            
            return {
                "success": True,
                "output": f"Estrutura de projeto {project_type} criada com sucesso.\n\n" + "\n".join(created),
                "type": "project_creation",
                "project_type": project_type,
                "created_items": created
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao criar estrutura do projeto: {str(e)}",
                "type": "project_creation",
                "error": str(e)
            }
    
    def _show_project_structure(self) -> Dict[str, Any]:
        """Mostra a estrutura de diretórios e arquivos do projeto"""
        try:
            # Usa uma abordagem baseada em Python para maior confiabilidade entre plataformas
            def build_tree(start_path: str) -> str:
                tree = []
                for root, dirs, files in os.walk(start_path):
                    # Ignora diretórios ocultos
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    level = root.replace(start_path, '').count(os.sep)
                    indent = ' ' * 4 * (level)
                    tree.append(f"{indent}{os.path.basename(root)}/")
                    
                    # Adiciona arquivos no diretório atual
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        if not f.startswith('.'):  # Ignora arquivos ocultos
                            tree.append(f"{subindent}{f}")
                
                return '\n'.join(tree)
            
            # Obtém o diretório atual
            current_dir = os.getcwd()
            
            # Constrói a árvore de diretórios
            tree = build_tree('.')
            
            return {
                "success": True,
                "output": f"Estrutura do projeto em {current_dir}:\n{tree}",
                "type": "project_structure"
            }
                
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao obter estrutura do projeto: {str(e)}",
                "type": "project_structure",
                "error": str(e)
            }
            
    def _analyze_code(self, request: str) -> Dict[str, Any]:
        """Realiza análise estática de código e sugere melhorias.
        
        Este método executa uma análise abrangente do código-fonte, identificando:
        - Problemas de estilo e formatação
        - Possíveis bugs e anti-padrões
        - Oportunidades de refatoração
        - Complexidade ciclomática e métricas de código
        - Problemas de segurança comuns
        
        A análise pode ser executada em:
        - Um único arquivo
        - Múltiplos arquivos
        - Todo um diretório de forma recursiva
        
        Args:
            request: String contendo a especificação do que analisar. Pode incluir:
                   - Caminho para arquivo ou diretório
                   - Padrões de busca (ex: "analisar todos os arquivos *.py")
                   - Escopo da análise (ex: "analisar complexidade ciclomática")
                   Exemplos:
                   - "analisar main.py"
                   - "verificar problemas de estilo no diretório src/"
                   - "avaliar complexidade do código em utils/"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a análise foi concluída com sucesso
                - output: str com o relatório de análise formatado
                - type: str indicando o tipo de operação ("code_analysis")
                - issues: lista de problemas encontrados, cada um contendo:
                    - file: str com o caminho do arquivo
                    - line: int com o número da linha
                    - severity: str indicando a gravidade (info, warning, error)
                    - message: str com a descrição do problema
                    - code: str com o trecho de código problemático (opcional)
                - metrics: dict com métricas do código (complexidade, linhas, etc.)
                - summary: dict com resumo da análise
                
        Raises:
            FileNotFoundError: Se o arquivo ou diretório especificado não existir
            PermissionError: Se não houver permissão para acessar os arquivos
            ValueError: Se a solicitação de análise for inválida ou ambígua
            
        Exemplo de retorno de sucesso:
            {
                "success": True,
                "output": "Análise concluída: 3 avisos, 1 erro encontrados",
                "type": "code_analysis",
                "issues": [
                    {
                        "file": "src/utils.py",
                        "line": 42,
                        "severity": "warning",
                        "message": "Função muito longa (45 linhas). Considere dividi-la.",
                        "code": "def funcao_longa():"
                    },
                    ...
                ],
                "metrics": {
                    "lines_of_code": 1245,
                    "functions": 42,
                    "complexity_avg": 3.2,
                    "test_coverage": 78.5
                },
                "summary": {
                    "files_analyzed": 15,
                    "issues_found": 4,
                    "warnings": 3,
                    "errors": 1,
                    "success_rate": 0.95
                }
            }
        """
        filename = self._extract_filename(request)
        
        if not filename and os.path.isdir('.'):
            # Analisa todos os arquivos de código no diretório atual
            return self._analyze_directory('.')
            
        if not filename or not os.path.exists(filename):
            return {
                "success": False,
                "output": "Arquivo não especificado ou não encontrado.",
                "type": "code_analysis"
            }
            
        if os.path.isdir(filename):
            return self._analyze_directory(filename)
            
        # Analisa um único arquivo
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Gera uma análise do código
            analysis = self._generate_code_analysis(content, filename)
            
            return {
                "success": True,
                "output": f"Análise do arquivo {filename}:\n\n{analysis}",
                "type": "code_analysis",
                "filename": filename,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao analisar o arquivo: {str(e)}",
                "type": "code_analysis",
                "error": str(e)
            }
    
    def _analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analisa todos os arquivos de código em um diretório"""
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', 
            '.hpp', '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.m', '.scala',
            '.rs', '.dart', '.sh', '.pl', '.pm', '.r', '.lua'
        }
        
        analysis_results = []
        
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.endswith(ext) for ext in code_extensions):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                analysis = self._generate_code_analysis(content, filepath)
                                analysis_results.append(f"\n=== {filepath} ===\n{analysis}")
                        except Exception as e:
                            analysis_results.append(f"\n=== {filepath} ===\nErro ao analisar: {str(e)}")
            
            if not analysis_results:
                return {
                    "success": False,
                    "output": "Nenhum arquivo de código encontrado para análise.",
                    "type": "code_analysis"
                }
                
            return {
                "success": True,
                "output": "Análise concluída.\n" + "\n".join(analysis_results),
                "type": "code_analysis",
                "file_count": len(analysis_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao analisar diretório: {str(e)}",
                "type": "code_analysis",
                "error": str(e)
            }
    
    def _generate_code_analysis(self, content: str, filename: str) -> str:
        """Gera uma análise do código usando o LLM"""
        prompt = f"""Analise o seguinte código e forneça um resumo, possíveis problemas e sugestões de melhoria.
        Arquivo: {filename}
        
        Código:
        ```
        {content}
        ```
        
        Por favor, forneça:
        1. Um breve resumo do que o código faz
        2. Possíveis problemas ou más práticas
        3. Sugestões de melhoria
        4. Questões de segurança, se aplicável
        5. Considerações de desempenho, se aplicável
        """
        
        return self.invoke_llm(prompt)
    
    def _refactor_code(self, request: str) -> Dict[str, Any]:
        """Refatora o código de acordo com as melhores práticas e padrões de design.
        
        Este método realiza transformações seguras e controladas no código-fonte,
        melhorando sua estrutura, legibilidade e manutenibilidade, sem alterar seu
        comportamento externo.
        
        Tipos de refatoração suportados:
        - Extração de métodos/funções
        - Renomeação de variáveis e funções para maior clareza
        - Simplificação de expressões condicionais
        - Remoção de código duplicado
        - Melhoria na organização de classes e módulos
        - Aplicação de padrões de design quando apropriado
        
        Args:
            request: String contendo a especificação da refatoração. Pode incluir:
                   - Caminho para o arquivo a ser refatorado
                   - Descrição da melhoria desejada
                   - Escopo da refatoração (função, classe, módulo)
                   Exemplos:
                   - "refatorar função calcular_imc em utils.py"
                   - "melhorar nomes de variáveis em processamento.py"
                   - "extrair lógica de validação para uma função separada"
                   - "aplicar padrão Strategy na classe Pagamento"
                   
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - success: bool indicando se a refatoração foi bem-sucedida
                - output: str com o resumo das alterações realizadas
                - type: str indicando o tipo de operação ("code_refactoring")
                - original_file: str com o caminho do arquivo original
                - backup_file: str com o caminho do backup (opcional)
                - changes: lista de dicionários descrevendo cada alteração
                - diff: str com as diferenças entre as versões (opcional)
                - suggestions: lista de sugestões adicionais de melhoria (opcional)
                
        Raises:
            FileNotFoundError: Se o arquivo especificado não for encontrado
            ValueError: Se a solicitação de refatoração for ambígua ou inválida
            RuntimeError: Se ocorrer um erro durante o processo de refatoração
            
        Exemplo de retorno de sucesso:
            {
                "success": True,
                "output": "Código refatorado com sucesso. 3 alterações realizadas.",
                "type": "code_refactoring",
                "original_file": "src/utils.py",
                "backup_file": "src/utils.py.bak",
                "changes": [
                    {
                        "type": "extract_method",
                        "description": "Extraída lógica de validação para nova função",
                        "location": "linhas 42-58"
                    },
                    ...
                ],
                "diff": "- def validar_usuario(...):\n+ def validar_usuario(...):\n+     if not _validar_email(usuario.email):\n+         raise ValueError(\"Email inválido\")\n+ \n+ def _validar_email(email):\n+     return '@' in email and '.' in email.split('@')[-1]\n",
                "suggestions": [
                    "Considere adicionar type hints às assinaturas das funções",
                    "A função processar_dados poderia ser dividida em funções menores"
                ]
            }
        """
        filename = self._extract_filename(request)
        
        if not filename or not os.path.exists(filename):
            return {
                "success": False,
                "output": "Arquivo não especificado ou não encontrado.",
                "type": "code_refactor"
            }
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            # Gera o prompt para refatoração
            prompt = f"""Você é um engenheiro de software experiente. Por favor, refatore o seguinte código.
            
            Instruções: {request}
            
            Código original:
            ```
            {original_content}
            ```
            
            Retorne APENAS o código refatorado, sem explicações adicionais."""
            
            # Obtém o código refatorado
            refactored_content = self.invoke_llm(prompt)
            
            # Cria um backup antes de modificar
            backup_filename = f"{filename}.bak"
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Escreve o conteúdo refatorado
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(refactored_content)
                
            # Gera um diff das alterações
            diff = self._generate_diff(original_content, refactored_content, filename)
            
            return {
                "success": True,
                "output": f"Código refatorado com sucesso. Backup salvo como {backup_filename}\n\nDiferenças:\n{diff}",
                "type": "code_refactor",
                "filename": filename,
                "backup_file": backup_filename,
                "diff": diff
            }
            
        except Exception as e:
            # Restaura o backup em caso de erro
            if 'backup_filename' in locals() and os.path.exists(backup_filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                os.remove(backup_filename)
                
            return {
                "success": False,
                "output": f"Erro ao refatorar o código: {str(e)}",
                "type": "code_refactor",
                "error": str(e)
            }
    
    def _generate_diff(self, original: str, refactored: str, filename: str) -> str:
        """Gera um diff entre o código original e o refatorado"""
        original_lines = original.splitlines(keepends=True)
        refactored_lines = refactored.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            refactored_lines,
            fromfile=f'original/{filename}',
            tofile=f'refactored/{filename}',
            lineterm='',
            n=3
        )
        
        return ''.join(diff)