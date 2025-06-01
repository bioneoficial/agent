import os
import re
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from agents.base_agent import BaseAgent

class FileAgent(BaseAgent):
    """
    Agente especializado em manipulação avançada de arquivos e geração de código.
    
    Funcionalidades:
    1. Criação de arquivos com templates inteligentes
    2. Edição de arquivos com análise de contexto
    3. Análise de código para sugestões de melhoria
    4. Geração de estruturas de projetos
    5. Refatoração inteligente
    """
    
    def __init__(self):
        system_prompt = """Você é um expert em desenvolvimento de software especializado em manipulação de arquivos e geração de código.

Suas responsabilidades:
1. Criar arquivos com código limpo e eficiente usando templates inteligentes
2. Editar arquivos existentes com compreensão profunda do contexto
3. Analisar código para sugerir melhorias e boas práticas
4. Gerar estruturas de projeto e arquivos de configuração
5. Refatorar código seguindo princípios SOLID e Clean Code

Ao gerar ou modificar código:
- Escreva código limpo, bem comentado e modular
- Siga as convenções e boas práticas da linguagem
- Inclua tratamento de erros apropriado
- Utilize design patterns quando relevante
- Garanta que o código seja testável e manutenível

IMPORTANTE: Retorne APENAS o código/conteúdo solicitado, sem explicações adicionais, sem delimitadores markdown."""
        
        super().__init__("FileAgent", system_prompt)
        
        # Mapeamento de extensões por linguagem
        self.lang_extensions = {
            'python': 'py', 'javascript': 'js', 'typescript': 'ts', 
            'java': 'java', 'c++': 'cpp', 'c#': 'cs', 'go': 'go',
            'rust': 'rs', 'ruby': 'rb', 'php': 'php', 'swift': 'swift',
            'kotlin': 'kt', 'scala': 'scala', 'r': 'r', 'matlab': 'm',
            'bash': 'sh', 'shell': 'sh', 'powershell': 'ps1', 'html': 'html',
            'css': 'css', 'sql': 'sql', 'markdown': 'md', 'yaml': 'yml',
            'json': 'json', 'xml': 'xml', 'dockerfile': ''
        }
        
        # Templates básicos por tipo de arquivo
        self.templates = self._load_templates()
        
        # Tipos de arquivo para agrupamento em projetos
        self.file_categories = {
            'source': ['py', 'js', 'ts', 'java', 'cpp', 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt'],
            'config': ['json', 'yml', 'yaml', 'toml', 'ini', 'conf', 'xml'],
            'docs': ['md', 'rst', 'txt', 'adoc'],
            'web': ['html', 'css', 'jsx', 'tsx', 'vue', 'svelte'],
            'data': ['csv', 'json', 'xml', 'sql', 'db'],
            'scripts': ['sh', 'bat', 'ps1']
        }
        
    def can_handle(self, request: str) -> bool:
        """Verifica se este agente pode lidar com a solicitação"""
        request_lower = request.lower()
        
        # Detectar explicitamente comandos git para rejeitar
        git_command_patterns = [
            r'\bgit\s+\w+', r'\bcommit\b', r'\bcommitar\b', r'\bcomitar\b',
            r'\bmerge\b', r'\bpull\b', r'\bpush\b', r'\bbranch\b', r'\bcheckout\b',
            r'\bversionar\b', r'\bstage\b', r'\badicionar\s+ao\s+commit\b'
        ]
        
        # Se parece claramente com um comando git, rejeitar
        if any(re.search(pattern, request_lower) for pattern in git_command_patterns):
            return False
            
        # Comandos explícitos de alta prioridade para FileAgent
        file_operations = [
            # Criação e edição de arquivos
            (r'criar\s+(um\s+)?(novo\s+)?arquivo\s+[\w\./]+', True),
            (r'criar\s+[\w\./]+\.(py|js|java|html|css|md|json|ts|cpp)', True),
            (r'(editar|modificar)\s+arquivo\s+[\w\./]+', True),
            (r'editar\s+[\w\./]+\.(py|js|java|html|css|md|json|ts|cpp)', True),
            
            # Análise e refatoração de código
            (r'analisar\s+código\s+', True), 
            (r'analisar\s+arquivo\s+', True),
            (r'análise\s+de\s+código', True),
            (r'refatorar\s+(código|arquivo)\s+', True),
            (r'refatorar\s+[\w\./]+\.(py|js|java|html|css|md|json|ts|cpp)', True),
            
            # Estruturas de projeto
            (r'criar\s+(um\s+)?(novo\s+)?projeto', True),
            (r'gerar\s+(uma\s+)?estrutura\s+de\s+projeto', True),
            (r'(criar|gerar)\s+projeto\s+[\w]+\s+do\s+tipo', True),
            
            # Palavras-chave específicas para operações de arquivo
            (r'\btemplate\b', True),
            (r'\bboilerplate\b', True),
            (r'\bscaffolding\b', True),
            (r'estrutura\s+de\s+diretórios', True),
            (r'\bestrutura\s+básica\b', True),
            
            # Comandos de criação de arquivos específicos por linguagem
            (r'criar\s+(uma\s+)?classe\s+', True),
            (r'criar\s+(uma\s+)?interface\s+', True),
            (r'criar\s+(um\s+)?módulo\s+', True),
            (r'criar\s+(uma\s+)?função\s+', True),
            
            # Comandos genéricos - menor prioridade
            (r'\barquivo\b.*\b(classe|método|função)\b', False),
            (r'\bcódigo\b.*\b(classe|método|função)\b', False)
        ]
        
        # Verificar padrões de operações de arquivo
        for pattern, is_high_priority in file_operations:
            if re.search(pattern, request_lower):
                return True
                
        # Verificar extensões de arquivo na solicitação - apenas se parece com uma operação de arquivo
        extension_pattern = re.search(r'\.(py|js|ts|java|cpp|cs|html|css|md|json|yml|xml)\b', request_lower)
        if extension_pattern and any(verb in request_lower for verb in ['criar', 'editar', 'modificar', 'gerar', 'analisar', 'refatorar']):
            return True
            
        # Verificar menções a linguagens de programação com contexto de operação
        for lang in self.lang_extensions.keys():
            if re.search(rf'(criar|editar|gerar|analisar|refatorar).+\b{lang}\b', request_lower):
                return True
        
        # Para solicitações relacionadas a métodos/funções em arquivos específicos
        if re.search(r'adicionar\s+(método|função|classe)\s+.+\s+em\s+.+\.(py|js|java)', request_lower):
            return True
            
        # Se não há padrões explícitos, rejeitar
        return False
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa solicitações relacionadas a arquivos e código"""
        request_lower = request.lower()
        
        # Determinar o tipo de operação
        if any(word in request_lower for word in ['criar', 'create', 'novo', 'new', 'gerar', 'generate']):
            if any(word in request_lower for word in ['projeto', 'project', 'estrutura', 'structure']):
                return self._create_project_structure(request)
            else:
                return self._create_file(request)
        elif any(word in request_lower for word in ['editar', 'edit', 'modificar', 'modify', 'alterar', 'change']):
            return self._edit_file(request)
        elif any(word in request_lower for word in ['analisar', 'analyze', 'revisar', 'review']):
            return self._analyze_code(request)
        elif any(word in request_lower for word in ['refatorar', 'refactor', 'melhorar', 'improve']):
            return self._refactor_code(request)
        else:
            # Default para criação de arquivo se não estiver claro
            return self._create_file(request)
    
    def _extract_filename(self, request: str) -> Optional[str]:
        """Extrai o nome do arquivo da solicitação"""
        # Buscar nomes de arquivo explícitos com extensão
        filename_match = re.search(r'(?:arquivo|file|chamado|named|nome)?\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,5})', request, re.IGNORECASE)
        if filename_match:
            return filename_match.group(1).strip()
        
        # Inferir da linguagem mencionada e gerar um nome
        for lang, ext in self.lang_extensions.items():
            if lang in request.lower():
                # Gerar nome descritivo com timestamp
                words = re.findall(r'\b[a-zA-Z]{4,}\b', request.lower())
                words = [w for w in words if w not in ['criar', 'create', 'arquivo', 'file', 'código', 'code', lang]]
                
                if words:
                    name = '_'.join(words[:2])  # Usar as duas primeiras palavras significativas
                else:
                    name = f"generated_{lang}"
                    
                timestamp = str(int(time.time()))[-4:]
                return f"{name}_{timestamp}.{ext}"
        
        return None
    
    def _create_file(self, request: str) -> Dict[str, Any]:
        """Cria um novo arquivo com conteúdo gerado inteligentemente"""
        # Extrair nome de arquivo
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o nome do arquivo. Por favor, especifique um nome com extensão (ex: 'criar arquivo exemplo.py')",
                "type": "file_creation"
            }
        
        # Verificar se o arquivo já existe
        path = Path(filename)
        if path.exists():
            return {
                "success": False,
                "output": f"O arquivo '{filename}' já existe. Use 'editar {filename}' para modificá-lo.",
                "type": "file_creation"
            }
        
        try:
            # Determinar o tipo de arquivo e linguagem
            ext = filename.split('.')[-1].lower()
            language = ext
            
            # Encontrar nome da linguagem para prompting mais preciso
            for lang, lang_ext in self.lang_extensions.items():
                if lang_ext == ext:
                    language = lang
                    break
            
            # Detectar possíveis templates
            template_type = self._detect_template_type(request, language)
            
            # Gerar conteúdo
            content = self._generate_file_content(request, filename, language, template_type)
            
            # Criar diretórios se necessário
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Escrever arquivo
            path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "output": f"Arquivo '{filename}' criado com sucesso ({len(content)} caracteres)",
                "type": "file_creation",
                "filename": filename,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao criar arquivo: {str(e)}",
                "type": "file_creation",
                "error": str(e)
            }
    
    def _generate_file_content(self, request: str, filename: str, language: str, template_type: str = None) -> str:
        """Gera conteúdo de arquivo com base na solicitação e template"""
        # Verificar se temos um template para este tipo
        template = self._get_template(language, template_type)
        
        # Construir prompt para o LLM
        prompt = f"""Gere código {language} para o seguinte pedido:

Pedido: {request}
Arquivo: {filename}
{"Template: " + template_type if template_type else ""}

Requisitos:
- Código completo e funcional
- Imports/dependências necessários
- Comentários úteis
- Seguir boas práticas de {language}
- Tratamento de erros adequado
- Código modular e bem organizado

{"Utilize esta estrutura como base:" if template else ""}
{template if template else ""}

Retorne APENAS o código, sem explicações adicionais:"""
        
        # Gerar código
        code = self.invoke_llm(prompt, temperature=0.4)
        
        # Sanitização adicional para código
        code = self.sanitize_llm_response(code)
        
        # Remover delimitadores markdown restantes
        code = re.sub(r'^```[a-zA-Z]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        
        return code
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Carrega templates básicos para diferentes linguagens e tipos de arquivo"""
        # Template simples para demonstração - em uma implementação real, 
        # poderíamos carregar de arquivos externos
        templates = {
            'python': {
                'basic': '#!/usr/bin/env python3\n\"\"\"\n[DESCRIÇÃO]\n\"\"\"\n\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()',
                'class': '#!/usr/bin/env python3\n\"\"\"\n[DESCRIÇÃO]\n\"\"\"\n\nclass [NOME]:\n    \"\"\"\n    [DOCUMENTAÇÃO]\n    \"\"\"\n    \n    def __init__(self):\n        pass',
                'api': 'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n',
                'script': '#!/usr/bin/env python3\n\"\"\"\n[DESCRIÇÃO]\n\"\"\"\n\nimport argparse\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description="[DESCRIÇÃO]")\n    return parser.parse_args()\n\ndef main():\n    args = parse_args()\n    # TODO: Implementar lógica\n\nif __name__ == "__main__":\n    main()',
                'test': 'import unittest\n\nclass Test[NOME](unittest.TestCase):\n    def setUp(self):\n        pass\n        \n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == "__main__":\n    unittest.main()'
            },
            'javascript': {
                'basic': '/**\n * [DESCRIÇÃO]\n */\n\nfunction main() {\n    // TODO: Implementar\n}\n\nmain();',
                'react': 'import React from "react";\n\nfunction [NOME]() {\n    return (\n        <div>\n            <h1>Hello World</h1>\n        </div>\n    );\n}\n\nexport default [NOME];',
                'node': '/**\n * [DESCRIÇÃO]\n */\n\nconst express = require("express");\nconst app = express();\nconst port = process.env.PORT || 3000;\n\napp.get("/", (req, res) => {\n    res.send("Hello World");\n});\n\napp.listen(port, () => {\n    console.log(`Server running on port ${port}`);\n});'
            }
        }
        
        return templates
    
    def _get_template(self, language: str, template_type: str = None) -> str:
        """Obtém o template apropriado para a linguagem e tipo"""
        if language in self.templates and template_type in self.templates[language]:
            return self.templates[language][template_type]
        elif language in self.templates:
            # Retornar template básico se o tipo específico não existir
            return self.templates[language].get('basic', '')
        else:
            return ''
    
    def _detect_template_type(self, request: str, language: str) -> Optional[str]:
        """Detecta o tipo de template a ser usado com base na solicitação"""
        request_lower = request.lower()
        
        # Mapeamento de templates por linguagem
        template_keywords = {
            'python': {
                'class': ['classe', 'class', 'objeto', 'object', 'oop'],
                'api': ['api', 'fastapi', 'flask', 'endpoint', 'rest'],
                'script': ['script', 'cli', 'comando', 'command', 'argparse'],
                'test': ['teste', 'test', 'unittest', 'pytest']
            },
            'javascript': {
                'react': ['react', 'componente', 'component', 'jsx', 'tsx', 'ui'],
                'node': ['node', 'express', 'servidor', 'server', 'api', 'rest']
            }
        }
        
        # Verificar palavras-chave para o idioma específico
        if language in template_keywords:
            for template_type, keywords in template_keywords[language].items():
                if any(keyword in request_lower for keyword in keywords):
                    return template_type
        
        # Default para template básico
        return 'basic'
    
    def _edit_file(self, request: str) -> Dict[str, Any]:
        """Edita um arquivo existente com análise de contexto"""
        # Extrair nome do arquivo
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o arquivo a ser editado. Por favor, especifique um nome de arquivo.",
                "type": "file_edit"
            }
        
        # Verificar se o arquivo existe
        path = Path(filename)
        if not path.exists():
            return {
                "success": False,
                "output": f"Arquivo '{filename}' não existe. Use 'criar {filename}' para criá-lo.",
                "type": "file_edit"
            }
        
        try:
            # Ler conteúdo atual
            current_content = path.read_text(encoding='utf-8')
            
            # Determinar a linguagem com base na extensão
            ext = filename.split('.')[-1].lower()
            language = ext
            for lang, lang_ext in self.lang_extensions.items():
                if lang_ext == ext:
                    language = lang
                    break
            
            # Analisar o contexto do código atual
            code_context = self._analyze_code_context(current_content, language)
            
            # Gerar prompt para edição inteligente
            prompt = f"""Edite o seguinte código {language} com base na solicitação:

Arquivo: {filename}
Solicitação de edição: {request}

Análise do código atual:
- Estrutura: {code_context.get('structure', 'Não disponível')}
- Principais elementos: {code_context.get('elements', 'Não disponíveis')}
- Padrões de design: {code_context.get('patterns', 'Não detectados')}

Conteúdo atual do arquivo:
```{language}
{current_content}
```

Requisitos para a edição:
- Mantenha a estrutura e estilo existentes
- Preserve comentários importantes
- Garanta que o código continue funcional
- Faça apenas as alterações necessárias
- Mantenha a mesma formatação

Retorne o código completo editado, não apenas as mudanças:"""
            
            # Obter conteúdo editado
            new_content = self.invoke_llm(prompt, temperature=0.3)
            new_content = self.sanitize_llm_response(new_content)
            
            # Remover delimitadores markdown
            new_content = re.sub(r'^```[a-zA-Z]*\n', '', new_content)
            new_content = re.sub(r'\n```$', '', new_content)
            
            # Escrever conteúdo atualizado
            path.write_text(new_content, encoding='utf-8')
            
            # Calcular diferenças
            changes = self._calculate_changes(current_content, new_content)
            
            return {
                "success": True,
                "output": f"Arquivo '{filename}' editado com sucesso ({changes['lines_changed']} linhas alteradas)",
                "type": "file_edit",
                "filename": filename,
                "changes": changes
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao editar arquivo: {str(e)}",
                "type": "file_edit",
                "error": str(e)
            }
    
    def _analyze_code_context(self, content: str, language: str) -> Dict[str, Any]:
        """Analisa o contexto do código para edição inteligente"""
        # Inicializar contexto padrão
        context = {
            'structure': 'Não detectada',
            'elements': [],
            'patterns': 'Não detectados'
        }
        
        # Análise básica com expressões regulares
        if language == 'python':
            # Detectar classes
            classes = re.findall(r'class\s+(\w+)', content)
            if classes:
                context['elements'].extend([f"Classe: {cls}" for cls in classes])
            
            # Detectar funções
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            if functions:
                context['elements'].extend([f"Função: {func}" for func in functions])
            
            # Detectar imports
            imports = re.findall(r'(?:import|from)\s+([\w\.]+)', content)
            if imports:
                context['elements'].extend([f"Import: {imp}" for imp in imports[:5]])
            
            # Detectar estrutura
            if classes and functions:
                context['structure'] = 'Orientada a objetos'
            elif functions and not classes:
                context['structure'] = 'Procedural'
            
        elif language in ['javascript', 'typescript']:
            # Detectar classes
            classes = re.findall(r'class\s+(\w+)', content)
            # Detectar funções
            functions = re.findall(r'function\s+(\w+)\s*\(', content)
            # Detectar métodos de classe
            methods = re.findall(r'(\w+)\s*\([^)]*\)\s*{', content)
            # Detectar imports
            imports = re.findall(r'(?:import|require)\s+([\w\.]+)', content)
            
            # Combinar elementos
            if classes:
                context['elements'].extend([f"Classe: {cls}" for cls in classes])
            if functions:
                context['elements'].extend([f"Função: {func}" for func in functions])
            if imports:
                context['elements'].extend([f"Import: {imp}" for imp in imports[:5]])
            
            # Detectar estrutura
            if 'React' in content or 'Component' in content:
                context['structure'] = 'React Component'
            elif 'express' in content.lower():
                context['structure'] = 'Express Server'
            elif classes:
                context['structure'] = 'Orientada a objetos'
            else:
                context['structure'] = 'Funcional/Procedural'
                
        # Formatamos os elementos como string para facilitar o uso no prompt
        context['elements'] = ', '.join(context['elements'][:10]) if context['elements'] else 'Não detectados'
        
        return context
    
    def _calculate_changes(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Calcula mudanças entre versões de arquivos"""
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        # Contar linhas adicionadas, removidas e modificadas
        added = len(new_lines) - len(old_lines) if len(new_lines) > len(old_lines) else 0
        removed = len(old_lines) - len(new_lines) if len(old_lines) > len(new_lines) else 0
        
        # Contar linhas modificadas (aproximação simples)
        modified = 0
        for i in range(min(len(old_lines), len(new_lines))):
            if old_lines[i] != new_lines[i]:
                modified += 1
        
        return {
            'lines_changed': added + removed + modified,
            'lines_added': added,
            'lines_removed': removed,
            'lines_modified': modified
        }
    
    def _analyze_code(self, request: str) -> Dict[str, Any]:
        """Analisa código para sugerir melhorias"""
        # Extrair nome do arquivo
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o arquivo a ser analisado.",
                "type": "code_analysis"
            }
        
        # Verificar se o arquivo existe
        path = Path(filename)
        if not path.exists():
            return {
                "success": False,
                "output": f"Arquivo '{filename}' não existe.",
                "type": "code_analysis"
            }
        
        try:
            # Ler conteúdo
            content = path.read_text(encoding='utf-8')
            
            # Determinar linguagem
            ext = filename.split('.')[-1].lower()
            language = ext
            for lang, lang_ext in self.lang_extensions.items():
                if lang_ext == ext:
                    language = lang
                    break
            
            # Gerar prompt para análise
            prompt = f"""Analise o seguinte código {language} e sugira melhorias:

Arquivo: {filename}

```{language}
{content}
```

Realize uma análise detalhada focando em:
1. Problemas de design e arquitetura
2. Oportunidades de refatoração
3. Potenciais bugs ou vulnerabilidades
4. Melhorias de desempenho
5. Legibilidade e manutenibilidade
6. Seguimento de boas práticas e padrões de {language}

Formato da resposta:
- Um breve resumo do código
- Lista de problemas encontrados
- Sugestões de melhorias
- Pontos positivos do código"""
            
            # Obter análise
            analysis = self.invoke_llm(prompt, temperature=0.3)
            
            return {
                "success": True,
                "output": analysis,
                "type": "code_analysis",
                "filename": filename
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao analisar código: {str(e)}",
                "type": "code_analysis",
                "error": str(e)
            }
    
    def _refactor_code(self, request: str) -> Dict[str, Any]:
        """Refatora código com base em princípios de Clean Code e SOLID"""
        # Extrair nome do arquivo
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Não foi possível determinar o arquivo a ser refatorado.",
                "type": "code_refactor"
            }
        
        # Verificar se o arquivo existe
        path = Path(filename)
        if not path.exists():
            return {
                "success": False,
                "output": f"Arquivo '{filename}' não existe.",
                "type": "code_refactor"
            }
        
        try:
            # Ler conteúdo
            current_content = path.read_text(encoding='utf-8')
            
            # Determinar linguagem
            ext = filename.split('.')[-1].lower()
            language = ext
            for lang, lang_ext in self.lang_extensions.items():
                if lang_ext == ext:
                    language = lang
                    break
            
            # Gerar prompt para refatoração
            prompt = f"""Refatore o seguinte código {language} para melhorar sua qualidade:

Arquivo: {filename}
Solicitação: {request}

```{language}
{current_content}
```

Princípios a serem aplicados:
- SOLID (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)
- DRY (Don't Repeat Yourself)
- KISS (Keep It Simple, Stupid)
- Nomes de variáveis e funções mais descritivos
- Decomposição de funções/métodos longos
- Redução de complexidade ciclomática
- Tratamento adequado de erros
- Melhoria da legibilidade geral

IMPORTANTE:
- Mantenha a funcionalidade exatamente igual
- Preserve comentários importantes
- Mantenha a compatibilidade com o código existente
- Documente as mudanças importantes com comentários

Retorne o código completamente refatorado:"""
            
            # Obter código refatorado
            new_content = self.invoke_llm(prompt, temperature=0.3)
            new_content = self.sanitize_llm_response(new_content)
            
            # Remover delimitadores markdown
            new_content = re.sub(r'^```[a-zA-Z]*\n', '', new_content)
            new_content = re.sub(r'\n```$', '', new_content)
            
            # Criar backup antes de substituir
            backup_path = str(path) + '.bak'
            Path(backup_path).write_text(current_content, encoding='utf-8')
            
            # Escrever conteúdo refatorado
            path.write_text(new_content, encoding='utf-8')
            
            # Calcular mudanças
            changes = self._calculate_changes(current_content, new_content)
            
            return {
                "success": True,
                "output": f"Arquivo '{filename}' refatorado com sucesso ({changes['lines_changed']} linhas alteradas). Backup criado em '{backup_path}'.",
                "type": "code_refactor",
                "filename": filename,
                "backup": backup_path,
                "changes": changes
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao refatorar código: {str(e)}",
                "type": "code_refactor",
                "error": str(e)
            }
    
    def _create_project_structure(self, request: str) -> Dict[str, Any]:
        """Cria uma estrutura de projeto com múltiplos arquivos"""
        # Extrair nome do projeto ou usar um nome padrão
        project_name_match = re.search(r'(?:projeto|project|estrutura|structure)\s+([a-zA-Z0-9_\-]+)', request, re.IGNORECASE)
        project_name = project_name_match.group(1) if project_name_match else f"project_{int(time.time())%10000}"
        
        # Determinar tipo de projeto da solicitação
        project_type = self._detect_project_type(request)
        
        try:
            # Gerar estrutura de arquivos com base no tipo de projeto
            files = self._generate_project_files(request, project_name, project_type)
            
            # Criar diretórios e arquivos
            for filepath, content in files.items():
                full_path = Path(f"{project_name}/{filepath}")
                os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else '.', exist_ok=True)
                full_path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "output": f"Estrutura de projeto '{project_name}' criada com sucesso ({len(files)} arquivos)",
                "type": "project_creation",
                "project_name": project_name,
                "files": list(files.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao criar estrutura de projeto: {str(e)}",
                "type": "project_creation",
                "error": str(e)
            }
    
    def _detect_project_type(self, request: str) -> str:
        """Detecta o tipo de projeto com base na solicitação"""
        request_lower = request.lower()
        
        # Mapeamento de tipos de projeto
        project_types = {
            'python_package': ['python', 'pacote', 'package', 'biblioteca', 'library'],
            'web_app': ['web', 'aplicação web', 'web app', 'site', 'html', 'css', 'javascript'],
            'api_rest': ['api', 'rest', 'restful', 'backend', 'servidor', 'server'],
            'frontend_react': ['react', 'frontend', 'interface', 'ui', 'spa'],
            'cli_tool': ['cli', 'command line', 'linha de comando', 'terminal', 'console']
        }
        
        # Verificar menções a tipos de projeto
        for project_type, keywords in project_types.items():
            if any(keyword in request_lower for keyword in keywords):
                return project_type
        
        # Default para pacote Python
        return 'python_package'
    
    def _generate_project_files(self, request: str, project_name: str, project_type: str) -> Dict[str, str]:
        """Gera conteúdo para os arquivos do projeto"""
        # Construir prompt para o LLM
        prompt = f"""Gere uma estrutura de projeto para a seguinte solicitação:

Solicitação: {request}
Nome do projeto: {project_name}
Tipo de projeto: {project_type}

Crie uma estrutura de arquivos completa e o conteúdo para cada arquivo.
Inclua arquivos de configuração, documentação e exemplos quando apropriado.
A estrutura deve seguir as melhores práticas para este tipo de projeto.

Formato da resposta:
{{
  "arquivos": [
    {{
      "caminho": "caminho/relativo/ao/arquivo.ext",
      "conteudo": "conteúdo completo do arquivo"
    }}
  ]
}}"""
        
        # Obter resposta do LLM
        response = self.invoke_llm(prompt, temperature=0.5)
        
        # Extrair estrutura de arquivos - fallback para estrutura básica em caso de erro
        try:
            # Tentar extrair JSON da resposta
            # Procurar estrutura JSON na resposta
            json_match = re.search(r'\{\s*"arquivos"\s*:\s*\[.+?\]\s*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                structure = json.loads(json_str)
                files = {item['caminho']: item['conteudo'] for item in structure['arquivos']}
            else:
                # Fallback: criar estrutura básica
                files = self._generate_basic_project_structure(project_name, project_type)
        except:
            # Fallback para estrutura básica em caso de erro
            files = self._generate_basic_project_structure(project_name, project_type)
        
        return files
    
    def _generate_basic_project_structure(self, project_name: str, project_type: str) -> Dict[str, str]:
        """Gera uma estrutura básica de projeto como fallback"""
        files = {}
        
        if project_type == 'python_package':
            # Estrutura básica de pacote Python
            files["README.md"] = f"# {project_name}\n\nDescrição do projeto."
            files["setup.py"] = f"from setuptools import setup, find_packages\n\nsetup(\n    name=\"{project_name}\",\n    version=\"0.1.0\",\n    packages=find_packages(),\n)"
            files["requirements.txt"] = "# Dependências do projeto"
            files[f"{project_name.lower()}/__init__.py"] = ""
            files[f"{project_name.lower()}/main.py"] = f"def main():\n    print(\"Hello from {project_name}\")\n\nif __name__ == \"__main__\":\n    main()"
            files["tests/__init__.py"] = ""
            files["tests/test_main.py"] = f"import unittest\nfrom {project_name.lower()}.main import main\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        # TODO: implementar teste\n        pass"
            
        elif project_type in ['web_app', 'frontend_react']:
            # Estrutura básica de aplicação web
            files["README.md"] = f"# {project_name}\n\nAplicação web."
            files["index.html"] = f"<!DOCTYPE html>\n<html>\n<head>\n    <title>{project_name}</title>\n    <link rel=\"stylesheet\" href=\"css/style.css\">\n</head>\n<body>\n    <h1>{project_name}</h1>\n    <script src=\"js/main.js\"></script>\n</body>\n</html>"
            files["css/style.css"] = "body {\n    font-family: Arial, sans-serif;\n    margin: 0;\n    padding: 20px;\n}"
            files["js/main.js"] = "// JavaScript principal\nconsole.log('Aplicação iniciada');"
            
        elif project_type == 'api_rest':
            # Estrutura básica de API REST
            files["README.md"] = f"# {project_name} API\n\nAPI REST."
            files["app.py"] = "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return jsonify({'message': 'Welcome to the API'})\n\nif __name__ == '__main__':\n    app.run(debug=True)"
            files["requirements.txt"] = "flask==2.0.1\n"
            files["tests/__init__.py"] = ""
            files["tests/test_api.py"] = "import unittest\nimport json\nfrom app import app\n\nclass TestAPI(unittest.TestCase):\n    def setUp(self):\n        self.app = app.test_client()\n\n    def test_home(self):\n        response = self.app.get('/')\n        data = json.loads(response.data)\n        self.assertEqual(response.status_code, 200)"
            
        elif project_type == 'cli_tool':
            # Estrutura básica de ferramenta CLI
            files["README.md"] = f"# {project_name}\n\nFerramenta de linha de comando."
            files["cli.py"] = "import argparse\n\ndef main():\n    parser = argparse.ArgumentParser(description='CLI tool')\n    parser.add_argument('--version', action='store_true', help='Show version')\n    args = parser.parse_args()\n    \n    if args.version:\n        print('v0.1.0')\n\nif __name__ == '__main__':\n    main()"
            files["requirements.txt"] = "# Dependências do projeto"
            
        return files
