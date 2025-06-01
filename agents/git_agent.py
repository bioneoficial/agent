import os
import subprocess
import shlex
from typing import Dict, Any, Optional, List, Tuple
from agents.base_agent import BaseAgent
import re

class GitAgent(BaseAgent):
    """Specialized agent for Git operations and commit message generation"""
    
    def __init__(self):
        system_prompt = """Você é um especialista em Git, especializado em gerar mensagens de commit semânticas precisas seguindo a especificação Conventional Commits.

# Formato da Mensagem de Commit
type(scope): description

# Tipos Válidos e Quando Usá-los
- feat: Nova funcionalidade, comportamento ou capacidade adicionada ao código
  - Exemplos: Novo endpoint de API, nova interface de usuário, novo algoritmo, nova validação
- fix: Correção de bug ou comportamento incorreto no código
  - Exemplos: Corrigir validação que falha, corrigir cálculo incorreto, resolver erro, consertar falha
- docs: Mudanças apenas na documentação (comentários, docstrings, README, etc.)
  - Exemplos: Atualizar instruções de instalação, documentar uma função, melhorar comentários
- style: Mudanças que não afetam o significado do código (formatação, espaços em branco)
  - Exemplos: Ajustar indentação, corrigir formatação, remover espaços extras, ajustar estilos
- refactor: Mudanças no código que não corrigem bugs nem adicionam recursos
  - Exemplos: Renomear variáveis, simplificar lógica, reorganizar código, melhorar estrutura
- test: Adicionando ou modificando testes
  - Exemplos: Adicionar teste unitário, melhorar teste de integração, corrigir testes falhos
- chore: Tarefas de manutenção, dependências, configuração
  - Exemplos: Atualizar dependências, configurar CI, ajustar configuração, organizar arquivos
- perf: Melhorias de desempenho
  - Exemplos: Otimizar algoritmo, melhorar consulta ao banco de dados, reduzir uso de memória
- ci: Mudanças em configurações de CI/CD
  - Exemplos: Configurar pipeline, ajustar regras de build, modificar fluxos de integração
- build: Sistema de build ou dependências externas
  - Exemplos: Atualizar sistema de compilação, modificar configurações de build

# Escopos Comuns e Seu Uso
- api: Alterações relacionadas a APIs, endpoints, serviços web
- ui: Alterações em interfaces de usuário, componentes visuais
- auth: Alterações relacionadas a autenticação e autorização
- data: Mudanças no acesso ou manipulação de dados
- core: Alterações no núcleo do sistema
- utils: Mudanças em utilitários ou helpers
- config: Alterações em arquivos de configuração
- deps: Mudanças em dependências
- security: Correções ou melhorias de segurança
- docs: Alterações em documentação (quando o tipo também é docs)
- tests: Alterações em testes (quando o tipo também é test)
- [linguagem]: Use o nome da linguagem (python, javascript, etc.) quando as mudanças são específicas

# Exemplos de Mensagens de Commit Excelentes
- feat(auth): implementar validação de tokens JWT
- fix(api): tratar resposta nula do serviço de usuários
- docs(readme): atualizar instruções de instalação
- style(formatter): aplicar indentação consistente
- refactor(utils): simplificar lógica de tratamento de erros
- test(login): adicionar testes unitários para validação de senha
- chore(deps): atualizar dependências para últimas versões
- perf(query): otimizar algoritmo de busca no banco de dados
- fix(security): corrigir vulnerabilidade XSS nos formulários
- feat(ui): adicionar componente de seleção de data

# Regras para Mensagens de Commit
1. Mantenha descrições concisas (menos de 72 caracteres)
2. Use modo imperativo ("adicionar" não "adicionado" ou "adiciona")
3. Seja específico sobre o que mudou e por quê
4. Primeira letra não é maiúscula
5. Sem ponto final
6. Foque no propósito da mudança, não nos detalhes técnicos
7. Se possível, mencione áreas afetadas na descrição
8. Use escopo quando a mudança afeta uma área específica

CRÍTICO: Retorne APENAS a linha da mensagem de commit. Sem explicações, sem formatação markdown.
Apenas retorne a mensagem de commit no formato: type(scope): description

Use as informações da análise do diff para gerar uma mensagem mais precisa. Priorize as categorias detectadas, funções e classes adicionadas, e palavras-chave relevantes."""
        
        super().__init__("GitAgent", system_prompt)
        
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the request"""
        request_lower = request.lower()
        git_keywords = [
            'git', 'commit', 'stage', 'add', 'push', 'pull', 'branch',
            'checkout', 'merge', 'rebase', 'status', 'diff', 'log',
            'commitar', 'comitar', 'adicionar', 'versionar'
        ]
        return any(keyword in request_lower for keyword in git_keywords)
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process Git-related requests"""
        request_lower = request.lower()
        
        # Determine the type of Git operation
        if any(word in request_lower for word in ['commit', 'commitar', 'comitar']):
            if any(word in request_lower for word in ['descrit', 'inteligente', 'auto', 'analise']):
                return self._generate_commit_message()
            else:
                return self._simple_commit()
        elif 'status' in request_lower:
            return self._git_status()
        elif 'diff' in request_lower:
            return self._git_diff()
        elif any(word in request_lower for word in ['add', 'adicionar', 'stage']):
            return self._git_add(request)
        else:
            return self._execute_git_command(request)
    
    def _safe_git_command(self, cmd: str) -> Dict[str, Any]:
        """Safely execute a git command with improved error handling"""
        try:
            # Validate the command is safe
            if any(dangerous in cmd.lower() for dangerous in ['--force', '-f', 'reset --hard', 'clean -fd']):
                return {
                    "success": False,
                    "output": f"⚠️ Command '{cmd}' may be destructive. Please be more specific or use Git directly.",
                    "type": "git_command",
                    "warning": "potentially_destructive"
                }
            
            # Run the git command
            result = subprocess.run(
                ["git"] + shlex.split(cmd),
                text=True,
                capture_output=True,
                cwd=os.getcwd()
            )
            
            success = result.returncode == 0
            output = result.stdout.strip() if success else result.stderr.strip()
            
            # Check for common error patterns and provide helpful responses
            if not success:
                error_output = result.stderr.lower()
                
                # Repository not initialized
                if "not a git repository" in error_output:
                    return {
                        "success": False,
                        "output": "This directory is not a Git repository. Initialize with 'git init' first.",
                        "type": "git_command",
                        "error_type": "not_a_repository"
                    }
                
                # No changes to commit
                elif "no changes added to commit" in error_output:
                    return {
                        "success": False,
                        "output": "No changes added to commit. Use 'git add <files>' first.",
                        "type": "git_command",
                        "error_type": "nothing_staged"
                    }
                
                # No commits yet
                elif "does not point to a valid object" in error_output or "no commit checked out" in error_output:
                    return {
                        "success": False,
                        "output": "Repository has no commits yet. Make your first commit to proceed.",
                        "type": "git_command",
                        "error_type": "no_commits"
                    }
                
                # Merge conflicts
                elif "merge conflict" in error_output or "needs merge" in error_output:
                    return {
                        "success": False,
                        "output": "There are merge conflicts that need to be resolved.",
                        "type": "git_command",
                        "error_type": "merge_conflict"
                    }
            
            return {
                "success": success,
                "output": output or "Command executed successfully",
                "type": "git_command",
                "command": cmd
            }
        except Exception as e:
            # Handle specific exceptions
            if isinstance(e, FileNotFoundError) and e.filename == 'git':
                return {
                    "success": False,
                    "output": "Git is not installed or not found in PATH.",
                    "type": "git_command",
                    "error_type": "git_not_found"
                }
            return {
                "success": False,
                "output": f"Error executing git command: {str(e)}",
                "type": "git_command",
                "error": str(e)
            }
    
    def _analyze_file_changes(self) -> Tuple[List[str], Dict[str, Any]]:
        """Analyze staged file changes to determine the type of changes"""
        # Get list of changed files
        files_result = self._safe_git_command("diff --cached --name-status")
        if not files_result["success"]:
            return [], {"error": "Failed to get file changes"}
            
        # Get the diff stats
        stats_result = self._safe_git_command("diff --cached --stat")
        
        # Categorize files by type
        file_categories = {
            "new": [],      # New files
            "modified": [], # Modified files
            "deleted": [],  # Deleted files
            "renamed": [],  # Renamed files
            "config": [],   # Configuration files
            "docs": [],     # Documentation files
            "tests": [],    # Test files
            "core": [],     # Core application files
            "ui": [],       # UI/frontend files
            "api": [],      # API-related files
            "db": [],       # Database-related files
            "security": [], # Security-related files
            "perf": []      # Performance-related files
        }
        
        change_stats = {
            "insertions": 0,
            "deletions": 0,
            "files_changed": 0
        }
        
        # Parse the file changes
        changed_files = []
        for line in files_result["output"].split('\n'):
            if not line.strip():
                continue
                
            status, *file_parts = line.split()
            file_path = ' '.join(file_parts)
            changed_files.append(file_path)
            
            # Categorize by status
            if status == 'A':
                file_categories["new"].append(file_path)
            elif status == 'M':
                file_categories["modified"].append(file_path)
            elif status == 'D':
                file_categories["deleted"].append(file_path)
            elif status.startswith('R'):
                file_categories["renamed"].append(file_path)
            
            # Categorize by file extension and path patterns
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path).lower()
            
            # Configuration files
            if any(file_path.endswith(ext) for ext in (
                '.json', '.yml', '.yaml', '.toml', '.ini', '.env', '.config', '.conf',
                'dockerfile', '.dockerignore', '.gitignore', '.eslintrc', '.prettierrc')):
                file_categories["config"].append(file_path)
            
            # Documentation files
            elif any(file_path.endswith(ext) for ext in (
                '.md', '.txt', '.rst', '.docx', '.pdf', '.wiki', '.adoc', '.epub')):
                file_categories["docs"].append(file_path)
            
            # Test files
            elif 'test' in file_path.lower() or file_path.startswith(('test/', 'tests/', 'spec/', '__tests__/')):
                file_categories["tests"].append(file_path)
            
            # UI/Frontend files
            elif any(file_path.endswith(ext) for ext in (
                '.css', '.scss', '.sass', '.less', '.html', '.vue', '.jsx', '.tsx',
                '.component.ts', '.component.js', '.svg', '.png', '.jpg', '.jpeg', '.gif')):
                file_categories["ui"].append(file_path)
            
            # API files
            elif any(('api' in part.lower() or 'endpoint' in part.lower() or 'route' in part.lower())
                    for part in file_path.split('/')) or file_path.endswith(('.route.js', '.controller.js', '.api.ts')):
                file_categories["api"].append(file_path)
            
            # Database files
            elif any(('db' in part.lower() or 'model' in part.lower() or 'schema' in part.lower() or 'migration' in part.lower())
                    for part in file_path.split('/')) or file_path.endswith(('.sql', '.prisma', '.sequelize', '.mongoose')):
                file_categories["db"].append(file_path)
                
            # Security files
            elif any(('auth' in part.lower() or 'security' in part.lower() or 'password' in part.lower() or 'crypto' in part.lower())
                    for part in file_path.split('/')):
                file_categories["security"].append(file_path)
                
            # Default to core files
            else:
                file_categories["core"].append(file_path)
        
        # Parse stats if available
        if stats_result["success"]:
            # Try to find the summary line with insertions/deletions
            for line in stats_result["output"].split('\n'):
                if 'file' in line and ('changed' in line or 'insertion' in line or 'deletion' in line):
                    parts = line.strip().split(', ')
                    for part in parts:
                        if 'file' in part and 'changed' in part:
                            try:
                                change_stats["files_changed"] = int(part.split()[0])
                            except ValueError:
                                pass
                        elif 'insertion' in part:
                            try:
                                change_stats["insertions"] = int(part.split()[0])
                            except ValueError:
                                pass
                        elif 'deletion' in part:
                            try:
                                change_stats["deletions"] = int(part.split()[0])
                            except ValueError:
                                pass
        
        return changed_files, {"categories": file_categories, "stats": change_stats}
    
    def _detect_file_language(self, file_path: str) -> str:
        """Detecta a linguagem de programação com base na extensão do arquivo"""
        ext = os.path.splitext(file_path)[1].lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'react',
            '.ts': 'typescript',
            '.tsx': 'react-typescript',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.sh': 'shell',
            '.md': 'markdown',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.xml': 'xml',
            '.sql': 'sql',
            '.graphql': 'graphql',
            '.dockerfile': 'dockerfile',
            '.vue': 'vue'
        }
        
        return language_map.get(ext, 'plaintext')
    
    def _suggest_commit_type(self, analysis) -> Tuple[str, str]:
        """Suggest a commit type and scope based on file analysis
        
        Returns:
            Tuple[str, str]: (commit_type, suggested_scope)
        """
        # Default to empty (will be determined by LLM)
        commit_type = ""
        suggested_scope = ""
        
        # File categories for decision making
        has_tests = len(analysis['categories']['tests']) > 0
        has_docs = len(analysis['categories']['docs']) > 0
        has_config = len(analysis['categories']['config']) > 0
        has_ui = len(analysis['categories']['ui']) > 0
        has_api = len(analysis['categories']['api']) > 0
        has_security = any('security' in file.lower() or 'auth' in file.lower() 
                          for file in analysis['categories']['modified'] + analysis['categories']['new'])
        
        # Check for test changes first
        if has_tests and len(analysis['categories']['modified']) + len(analysis['categories']['new']) == len(analysis['categories']['tests']):
            commit_type = "test"
            # See if tests are focused on a specific area
            test_dirs = set()
            for test_file in analysis['categories']['tests']:
                parts = test_file.split('/')
                if len(parts) > 2 and parts[0] == 'tests':
                    test_dirs.add(parts[1])
            if len(test_dirs) == 1:
                suggested_scope = next(iter(test_dirs))
        
        # Check for documentation-only changes
        elif has_docs and len(analysis['categories']['modified']) + len(analysis['categories']['new']) == len(analysis['categories']['docs']):
            commit_type = "docs"
            # If only README changed
            if len(analysis['categories']['docs']) == 1 and 'README' in analysis['categories']['docs'][0]:
                suggested_scope = "readme"
        
        # Check for configuration/setup changes
        elif has_config and len(analysis['categories']['modified']) + len(analysis['categories']['new']) == len(analysis['categories']['config']):
            commit_type = "chore"
            suggested_scope = "config"
        
        # Check for security-related changes
        elif has_security:
            if len(analysis['categories']['new']) > len(analysis['categories']['modified']):
                commit_type = "feat"
            else:
                commit_type = "fix"
                if analysis['categories']["new"] and not analysis['categories']["modified"]:
                    commit_type = "feat"
                elif analysis['categories']["modified"] and not analysis['categories']["new"]:
                    commit_type = "fix"
                suggested_scope = "api"
        
        elif analysis['categories']["security"] and len(analysis['categories']["security"]) > 0:
            commit_type = "fix" 
            suggested_scope = "security"
        
        elif analysis['categories']["db"] and len(analysis['categories']["db"]) > 0:
            if analysis['categories']["new"] and not analysis['categories']["modified"]:
                commit_type = "feat"
            else:
                commit_type = "fix"
            suggested_scope = "db"
        
        # New files typically indicate features
        elif analysis['categories']["new"] and len(analysis['categories']["new"]) > 0 and not analysis['categories']["modified"]:
            commit_type = "feat"
            # Try to derive scope from common parent directory
            if len(analysis['categories']["new"]) > 1:
                paths = [p.split('/') for p in analysis['categories']["new"]]
                if all(len(p) > 1 for p in paths):
                    common_parent = paths[0][0]
                    if all(p[0] == common_parent for p in paths):
                        suggested_scope = common_parent
        
        # Deletions with few insertions might be refactoring or cleanup
        elif analysis['stats']["deletions"] > 3 * analysis['stats']["insertions"] and analysis['stats']["insertions"] > 0:
            commit_type = "refactor"
        
        # For mixed changes, we'll let the LLM decide based on diff content
        return commit_type, suggested_scope
    
    def _extract_diff_content(self, changed_files: List[str], max_lines: int = 50) -> Dict[str, Any]:
        """Extract detailed semantic information from the diff content
        
        Returns a dictionary with diff samples and semantic analysis data.
        """
        diff_samples = []  # Sample diff content for LLM context
        total_lines = 0
        
        # Semantic analysis containers
        imports_added = []
        functions_added = []
        functions_modified = []
        classes_added = []
        classes_modified = []
        keywords = set()
        docstring_changes = False
        security_related = False
        api_related = False
        ui_related = False
        test_related = False
        performance_related = False
        file_languages = {}  # Armazena a linguagem de cada arquivo modificado
        
        # Security-related patterns (expandidos)
        security_patterns = [
            r'password', r'auth', r'token', r'crypt', r'hash', r'secret', 
            r'security', r'vulnerability', r'exploit', r'permission', r'login',
            r'senha', r'autenticação', r'autorização', r'credencial', r'cert',
            r'verify', r'validation', r'sanitize', r'escape', r'encoding',
            r'decrypt', r'encrypt', r'sign', r'verif', r'attack', r'threat',
            r'CVE', r'injection', r'XSS', r'CSRF', r'CORS', r'firewall'
        ]
        
        # API-related patterns (expandidos)
        api_patterns = [
            r'api', r'endpoint', r'route', r'request', r'response', r'http', 
            r'rest', r'get', r'post', r'put', r'delete', r'payload', r'header',
            r'param', r'query', r'body', r'content', r'json', r'xml', r'status',
            r'code', r'(2|4|5)[0-9]{2}', r'controller', r'service', r'resource',
            r'graphql', r'grpc', r'websocket', r'webhook', r'callback', r'async',
            r'fetch', r'axios', r'client', r'server', r'backend', r'frontend'
        ]
        
        # UI-related patterns (expandidos)
        ui_patterns = [
            r'component', r'render', r'view', r'ui', r'css', r'style', r'html', 
            r'display', r'button', r'form', r'template', r'interface', r'screen',
            r'página', r'page', r'layout', r'theme', r'cor', r'color', r'font',
            r'responsive', r'mobile', r'desktop', r'flex', r'grid', r'animation',
            r'transition', r'modal', r'dialog', r'popup', r'alert', r'notification',
            r'toast', r'menu', r'nav', r'tab', r'card', r'badge', r'tooltip',
            r'slider', r'scroll', r'drag', r'drop', r'swipe', r'gesture'
        ]
        
        # Performance-related patterns (expandidos)
        perf_patterns = [
            r'performance', r'optimize', r'speed', r'memory', r'efficient', 
            r'fast', r'slow', r'bottleneck', r'cache', r'timeout', r'lazy',
            r'eager', r'render', r'loading', r'parsing', r'throughput', r'latency',
            r'response time', r'processing', r'execute', r'runtime', r'memory leak',
            r'garbage', r'collect', r'allocation', r'buffer', r'stream', r'chunk',
            r'batch', r'throttle', r'debounce', r'measure', r'profile', r'benchmark',
            r'O\([n0-9]+\)', r'complexidade', r'complexity', r'overhead'
        ]
        
        # Test-related patterns (expandidos)
        test_patterns = [
            r'test', r'spec', r'assert', r'mock', r'stub', r'fixture', r'expect',
            r'should', r'describe', r'it\s*\(', r'suite', r'scenario', r'given',
            r'when', r'then', r'verify', r'validate', r'unit', r'integration',
            r'e2e', r'end-to-end', r'smoke', r'regression', r'performance',
            r'load', r'stress', r'coverage', r'jest', r'pytest', r'unittest',
            r'testcase', r'beforeeach', r'aftereach', r'setup', r'teardown'
        ]
        
        # Padrões para refatoração
        refactor_patterns = [
            r'refactor', r'clean', r'improve', r'simplif', r'restructur',
            r'reorganiz', r'renam', r'mov', r'extract', r'split', r'merg',
            r'rewrite', r'better', r'reduc', r'remov', r'dead code',
            r'unused', r'duplica', r'redundant', r'legacy'
        ]
        
        # Limit files to analyze to avoid excessive prompt size
        for file_path in changed_files[:5]:  # Only analyze first 5 files
            try:
                # Detect file language and store it
                file_language = self._detect_file_language(file_path)
                file_languages[file_path] = file_language
                
                # Get the diff for staged changes
                result = subprocess.run(
                    ["git", "diff", "--staged", "--", file_path],
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                file_diff = result.stdout.strip()
                if not file_diff:
                    continue
                    
                # Extract a subset of the diff for context
                lines = file_diff.split('\n')
                
                # Create a meaningful sample for the LLM
                header_lines = [line for line in lines[:5] if line.startswith('+++') or line.startswith('---')]
                
                # Extract added/removed lines
                added_lines = [line[1:] for line in lines if line.startswith('+') and not line.startswith('+++')]
                removed_lines = [line[1:] for line in lines if line.startswith('-') and not line.startswith('---')]
                
                # Analyze content for semantic meaning
                all_changed_lines = added_lines + removed_lines
                content_text = '\n'.join(all_changed_lines)
                
                # Detect imports
                for line in added_lines:
                    if re.match(r'^\s*(import|from)\s+\w+', line):
                        imports_added.append(line.strip())
                
                # Detect function definitions
                for line in added_lines:
                    func_match = re.match(r'^\s*def\s+(\w+)', line)
                    if func_match:
                        functions_added.append(func_match.group(1))
                
                # Detect function modifications
                for i, line in enumerate(removed_lines):
                    func_match = re.match(r'^\s*def\s+(\w+)', line)
                    if func_match:
                        func_name = func_match.group(1)
                        # Check if this function was modified rather than deleted
                        if any(re.match(r'^\s*def\s+' + re.escape(func_name), a) for a in added_lines):
                            functions_modified.append(func_name)
                
                # Detect class definitions
                for line in added_lines:
                    class_match = re.match(r'^\s*class\s+(\w+)', line)
                    if class_match:
                        classes_added.append(class_match.group(1))
                
                # Detect docstring changes
                docstring_pattern = r'("""|\'\'\')'
                if re.search(docstring_pattern, content_text):
                    docstring_changes = True
                
                # Keyword extraction (collect significant words for commit type inference)
                words = re.findall(r'\b\w+\b', content_text.lower())
                significant_words = [w for w in words if len(w) > 3 and w not in [
                    'self', 'this', 'that', 'with', 'from', 'have', 'been',
                    'were', 'they', 'will', 'would', 'should', 'could', 'than',
                    'then', 'when', 'what', 'where', 'which', 'there', 'their',
                    'return', 'import', 'class', 'function', 'method', 'print',
                    'none', 'true', 'false', 'pass', 'continue', 'break'
                ]]
                keywords.update(significant_words)
                
                # Check for refactoring-related changes
                refactor_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                     for pattern in refactor_patterns)
                
                # Check for security-related changes
                if not security_related:
                    security_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                          for pattern in security_patterns)
                
                # Check for API-related changes
                if not api_related:
                    api_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                     for pattern in api_patterns)
                
                # Check for UI-related changes
                if not ui_related:
                    ui_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                    for pattern in ui_patterns)
                
                # Check for test-related changes
                if not test_related:
                    test_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                      for pattern in test_patterns)
                
                # Check for performance-related changes
                if not performance_related:
                    performance_related = any(re.search(pattern, content_text, re.IGNORECASE) 
                                            for pattern in perf_patterns)
                
                # Análise específica para a linguagem de programação
                if file_language == 'python':
                    # Procurar por importações específicas do Python
                    for line in added_lines:
                        if re.match(r'^\s*(import|from)\s+\w+', line):
                            imports_added.append(line.strip())
                    
                    # Análise de funções e classes específicas para Python
                    for line in added_lines:
                        func_match = re.match(r'^\s*def\s+(\w+)', line)
                        if func_match:
                            functions_added.append(func_match.group(1))
                        
                        class_match = re.match(r'^\s*class\s+(\w+)', line)
                        if class_match:
                            classes_added.append(class_match.group(1))
                
                elif file_language in ['javascript', 'typescript', 'react', 'react-typescript']:
                    # Procurar por importações específicas do JS/TS
                    for line in added_lines:
                        if re.search(r'(import|require)\s+[\w{},\s]+\s+from', line):
                            imports_added.append(line.strip())
                    
                    # Análise de funções e classes específicas para JS/TS
                    for line in added_lines:
                        # Funções JS/TS (várias formas de declaração)
                        func_match = re.search(r'(function\s+(\w+)|const\s+(\w+)\s*=\s*\(|\s*(\w+)\s*:\s*function|\s*(\w+)\s*=\s*\(.*\)\s*=>)', line)
                        if func_match:
                            groups = func_match.groups()
                            function_name = next((g for g in groups if g is not None and g != 'function'), None)
                            if function_name:
                                functions_added.append(function_name)
                        
                        # Classes JS/TS
                        class_match = re.search(r'class\s+(\w+)', line)
                        if class_match:
                            classes_added.append(class_match.group(1))
                
                # Take a balanced sample of added and removed lines for LLM context
                max_sample_lines = 15
                added_sample = added_lines[:max_sample_lines//2]
                removed_sample = removed_lines[:max_sample_lines//2]
                
                # Create a formatted sample with headers
                file_sample = (header_lines + 
                               [f"--------- Arquivo: {file_path} ({file_language}) ---------"] +
                               ["--------- Linhas Removidas ---------"] + ["-" + line for line in removed_sample] + 
                               ["--------- Linhas Adicionadas ---------"] + ["+" + line for line in added_sample])
                
                if file_sample:
                    diff_samples.append('\n'.join(file_sample))
                    total_lines += len(file_sample)
                    
                    # Break if we've collected enough lines
                    if total_lines >= max_lines:
                        break
            except subprocess.CalledProcessError:
                pass
        
        return {
            "diff_samples": diff_samples,
            "imports_added": imports_added,
            "functions_added": functions_added,
            "functions_modified": functions_modified,
            "classes_added": classes_added,
            "docstring_changes": docstring_changes,
            "keywords": list(keywords)[:20],  # Limit to top 20 keywords
            "security_related": security_related,
            "api_related": api_related,
            "ui_related": ui_related,
            "test_related": test_related,
            "performance_related": performance_related,
            "refactor_related": refactor_related,
            "file_languages": file_languages
        }
    
    def _get_diff_keywords(self, diff_output: str) -> List[str]:
        """Extract keywords from diff content to better understand changes"""
        # Look for specific patterns in diff that might indicate intent
        keywords = []
        
        # Look for bug fix related terms
        if any(term in diff_output.lower() for term in [
                'fix', 'bug', 'issue', 'error', 'crash', 'exception', 'fail', 'resolve',
                'problem', 'incorrect', 'invalid', 'wrong']):
            keywords.append('fix')
            
        # Look for feature related terms
        if any(term in diff_output.lower() for term in [
                'feature', 'add', 'new', 'implement', 'support', 'create', 'introduce']):
            keywords.append('feature')
            
        # Look for performance related terms
        if any(term in diff_output.lower() for term in [
                'performance', 'optimize', 'speed', 'fast', 'slow', 'memory', 'efficient',
                'bottleneck', 'latency']):
            keywords.append('performance')
            
        # Look for refactoring related terms
        if any(term in diff_output.lower() for term in [
                'refactor', 'clean', 'simplify', 'restructure', 'improve', 'clarity',
                'maintainability', 'readability']):
            keywords.append('refactor')
            
        # Look for security related terms
        if any(term in diff_output.lower() for term in [
                'security', 'vulnerability', 'secure', 'auth', 'protect', 'encrypt',
                'safety', 'injection', 'xss', 'csrf']):
            keywords.append('security')
            
        return keywords
    
    def _generate_commit_message(self) -> Dict[str, Any]:
        """Generate an intelligent commit message based on staged changes"""
        # Get diff information for complete context
        diff_result = self._safe_git_command("diff --cached")
        if not diff_result["success"] or not diff_result["output"]:
            return {
                "success": False,
                "output": "No staged changes to generate commit message",
                "type": "commit_message"
            }
        
        # Analyze file changes
        changed_files, file_analysis = self._analyze_file_changes()
        if not changed_files:
            return {
                "success": False,
                "output": "No staged changes to generate commit message",
                "type": "commit_message"
            }
        
        # Get detailed diff content analysis
        diff_analysis = self._extract_diff_content(changed_files)
        
        # Get suggested commit type and scope
        suggested_type, suggested_scope = self._suggest_commit_type(file_analysis)
        
        # Refine type and scope based on diff content analysis
        refined_type, refined_scope = self._refine_commit_type_from_diff(diff_analysis)
        
        # Mostrar o tipo e escopo com base em análise de arquivos e análise semântica do diff
        final_type = refined_type or suggested_type or ""
        final_scope = refined_scope or suggested_scope or ""
        
        # Preparar exemplos de código mais relevantes para o contexto
        code_samples = self._prepare_relevant_code_samples(diff_analysis)
        
        # Prepare prompt for LLM with rich context and structure
        prompt = f"""Generate a single semantic commit message for these changes following the Conventional Commits specification.

Summary of Changes:
- Files changed: {file_analysis['stats']['files_changed']}
- Insertions: {file_analysis['stats']['insertions']}
- Deletions: {file_analysis['stats']['deletions']}

File Categories:
{', '.join([f"{cat}: {len(files)}" for cat, files in file_analysis['categories'].items() if files])}

Changed Files:
{', '.join([os.path.basename(f) for f in changed_files[:5]])}{' and more...' if len(changed_files) > 5 else ''}

Code Analysis:
{self._format_diff_analysis_for_prompt(diff_analysis)}

Relevant Code Changes:
{code_samples}

## Guidance
{f'Tipo de commit sugerido: {final_type}' if final_type else 'Escolha o tipo de commit mais apropriado (feat, fix, docs, style, refactor, test, chore, perf, ci, build)'}
{f'Escopo sugerido: {final_scope}' if final_scope else 'Determine um escopo apropriado se necessário'}

Dica: A análise de código acima já indica se as mudanças estão relacionadas a novas funcionalidades, correções, refatoração, etc. Use essas informações para escolher o tipo e escopo corretos.

Baseado nas informações acima, gere uma mensagem de commit semântica concisa seguindo o formato Conventional Commits.
Retorne APENAS a mensagem de commit no formato: type(scope): description
"""
        
        # Generate commit message using LLM
        try:
            response = self.llm.invoke(prompt).content
            message = response.strip()
            
            # Ensure message follows conventional commits format
            message = self._fix_commit_format(message)
            
            # Se não conseguimos inferir o tipo/escopo antes, mas o LLM conseguiu,
            # vamos usá-los para futura análise
            if not final_type or not final_scope:
                match = re.match(r'^(\w+)(\([\w-]+\))?:', message)
                if match:
                    message_type = match.group(1)
                    message_scope = match.group(2)[1:-1] if match.group(2) else None
                    
                    if not final_type and message_type:
                        final_type = message_type
                    if not final_scope and message_scope:
                        final_scope = message_scope
            
            return {
                "success": True,
                "output": message,
                "type": "commit_message",
                "generated": True,
                "analysis": {
                    "file_analysis": file_analysis,
                    "diff_analysis": diff_analysis,
                    "suggested_type": final_type,
                    "suggested_scope": final_scope
                }
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Failed to generate commit message: {str(e)}",
                "type": "commit_message",
                "error": str(e)
            }
    
    def _prepare_relevant_code_samples(self, diff_analysis: Dict[str, Any]) -> str:
        """Prepare relevant code samples based on the diff analysis"""
        samples = []
        
        # Priorizar amostras mais relevantes
        if diff_analysis.get('diff_samples'):
            # Se temos funções adicionadas, priorizar amostras dessas funções
            if diff_analysis.get('functions_added') and len(diff_analysis['diff_samples']) > 1:
                for sample in diff_analysis['diff_samples']:
                    if any(func in sample for func in diff_analysis['functions_added']):
                        samples.append(sample)
                        if len(samples) >= 2:  # Limitar a 2 amostras relevantes
                            break
            
            # Se temos classes adicionadas e ainda precisamos de amostras
            if diff_analysis.get('classes_added') and len(samples) < 2:
                for sample in diff_analysis['diff_samples']:
                    if any(cls in sample for cls in diff_analysis['classes_added']):
                        if sample not in samples:  # Evitar duplicação
                            samples.append(sample)
                            if len(samples) >= 2:
                                break
            
            # Se ainda não temos amostras suficientes, adicionar do conjunto geral
            if not samples and diff_analysis['diff_samples']:
                samples = diff_analysis['diff_samples'][:2]
        
        # Se ainda não temos amostras, retornar mensagem informativa
        if not samples:
            return "No relevant code samples available"
            
        return "\n\n".join(samples)
    
    def _format_diff_analysis_for_prompt(self, diff_analysis: Dict[str, Any]) -> str:
        """Format the diff analysis data for inclusion in the LLM prompt"""
        parts = []
        
        # Add file languages information
        if diff_analysis.get("file_languages"):
            language_counts = {}
            for lang in diff_analysis["file_languages"].values():
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            parts.append("Linguagens detectadas:")
            for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
                parts.append(f"- {lang}: {count} arquivo(s)")
            parts.append("")
        
        # Add imports if present
        if diff_analysis.get("imports_added"):
            parts.append("Imports adicionados:")
            for imp in diff_analysis["imports_added"][:5]:  # Limit to 5 imports
                parts.append(f"- {imp}")
            parts.append("")
        
        # Add functions if present
        if diff_analysis.get("functions_added"):
            parts.append("Funções adicionadas:")
            for func in diff_analysis["functions_added"][:5]:  # Limit to 5 functions
                parts.append(f"- {func}")
            parts.append("")
        
        if diff_analysis.get("functions_modified"):
            parts.append("Funções modificadas:")
            for func in diff_analysis["functions_modified"][:5]:  # Limit to 5 functions
                parts.append(f"- {func}")
            parts.append("")
            
        # Add classes if present
        if diff_analysis.get("classes_added"):
            parts.append("Classes adicionadas:")
            for cls in diff_analysis["classes_added"][:5]:  # Limit to 5 classes
                parts.append(f"- {cls}")
            parts.append("")
            
        # Note docstring changes
        if diff_analysis.get("docstring_changes"):
            parts.append("Documentação modificada (docstrings)")
            parts.append("")
            
        # Add keywords
        if diff_analysis.get("keywords"):
            parts.append("Palavras-chave relevantes:")
            parts.append(", ".join(diff_analysis["keywords"]))
            parts.append("")
            
        # Add category information
        categories = []
        if diff_analysis.get("security_related"):
            categories.append("Segurança")
        if diff_analysis.get("api_related"):
            categories.append("API")
        if diff_analysis.get("ui_related"):
            categories.append("UI")
        if diff_analysis.get("test_related"):
            categories.append("Testes")
        if diff_analysis.get("performance_related"):
            categories.append("Performance")
        if diff_analysis.get("refactor_related"):
            categories.append("Refatoração")
            
        if categories:
            parts.append("Categorias detectadas: " + ", ".join(categories))
            parts.append("")
        
        # Sugestão de tipo e escopo
        commit_type, scope = self._refine_commit_type_from_diff(diff_analysis)
        if commit_type:
            parts.append(f"Tipo de commit sugerido: {commit_type}")
        if scope:
            parts.append(f"Escopo sugerido: {scope}")
        if commit_type or scope:
            parts.append("")
            
        # Add diff samples
        if diff_analysis.get("diff_samples"):
            parts.append("Amostra das alterações:")
            parts.append("")
            # Just include the first sample to avoid making the prompt too large
            parts.append(diff_analysis["diff_samples"][0])
            
        return "\n".join(parts)
    
    def _refine_commit_type_from_diff(self, diff_analysis: Dict[str, Any]) -> Tuple[str, str]:
        """Refine commit type and scope based on diff content analysis"""
        refined_type = None
        refined_scope = None
        
        # Inferir tipo de commit com base na análise do diff
        
        # Alterações apenas em documentação
        if diff_analysis.get("docstring_changes") and not diff_analysis.get("functions_added") and not diff_analysis.get("classes_added"):
            refined_type = "docs"
            
        # Mudanças relacionadas a testes
        elif diff_analysis.get("test_related"):
            refined_type = "test"
            
        # Alterações de segurança
        elif diff_analysis.get("security_related"):
            if diff_analysis.get("functions_added") or diff_analysis.get("classes_added"):
                refined_type = "feat"
            else:
                refined_type = "fix"
            refined_scope = "security"
                
        # Melhorias de performance
        elif diff_analysis.get("performance_related"):
            refined_type = "perf"
            
        # Refatoração de código
        elif diff_analysis.get("refactor_related") and not (diff_analysis.get("functions_added") or diff_analysis.get("classes_added")):
            refined_type = "refactor"
            
        # Novas funções ou classes (provavelmente novas features)
        elif diff_analysis.get("functions_added") or diff_analysis.get("classes_added"):
            refined_type = "feat"
                
        # Se não identificamos o tipo ainda, vamos analisar o escopo
        if not refined_scope:
            # Escopo baseado em linguagens predominantes
            if diff_analysis.get("file_languages"):
                lang_counts = {}
                for lang in diff_analysis["file_languages"].values():
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    
                # Se temos mais de 80% dos arquivos em uma linguagem específica
                total_files = sum(lang_counts.values())
                for lang, count in lang_counts.items():
                    if count / total_files > 0.8 and lang != 'plaintext':
                        refined_scope = lang
                        break
            
            # Alterações na UI
            if diff_analysis.get("ui_related") and not refined_scope:
                refined_scope = "ui"
                
            # Alterações na API
            elif diff_analysis.get("api_related") and not refined_scope:
                refined_scope = "api"
        
        # Se ainda não determinamos o tipo, verificar palavras-chave
        if not refined_type and diff_analysis.get("keywords"):
            keywords = set(k.lower() for k in diff_analysis["keywords"])
            
            # Palavras-chave que sugerem correções
            if any(kw in keywords for kw in ["fix", "bug", "issue", "error", "crash", "problema", "corrigir", "resolve", "resolve"]):
                refined_type = "fix"
                
            # Palavras-chave que sugerem novas funcionalidades
            elif any(kw in keywords for kw in ["add", "adiciona", "implementa", "nova", "novo", "feature", "funcionalidade", "criar", "create", "support"]):
                refined_type = "feat"
                
            # Palavras-chave que sugerem refatoração
            elif any(kw in keywords for kw in ["refator", "clean", "limpa", "melhora", "improve", "simplifica", "restructura", "reorganiza"]):
                refined_type = "refactor"
                
            # Palavras-chave que sugerem mudanças de estilo
            elif any(kw in keywords for kw in ["style", "estilo", "format", "formata", "lint", "prettier", "eslint", "indentation", "identação"]):
                refined_type = "style"
                
        # Padrão: se ainda não temos um tipo, usar "chore"
        if not refined_type:
            refined_type = "chore"
            
        return refined_type, refined_scope
        
    def _fix_commit_format(self, message: str) -> str:
        """Try to fix common issues with commit message format
        
        This function takes a commit message and attempts to format it according to
        Conventional Commits standard if it doesn't already conform.
        """
        # Common commit types
        valid_types = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'perf', 'ci', 'build']
        
        # Pattern to match a conventional commit
        conv_commit_pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\([\w-]+\))?:\s*.+'
        
        # If already valid, just clean it up
        if re.match(conv_commit_pattern, message.lower()):
            # Ensure no capital at beginning of description and no period at end
            return self._clean_commit_message(message)
        
        # Check if we have a type but it's malformatted (missing colon, etc.)
        for commit_type in valid_types:
            # Check for patterns like "feat - add new feature" or "feat add new feature"
            type_pattern = f'^{commit_type}\\s*[-–—]?\\s+'
            match = re.match(type_pattern, message.lower())
            if match:
                prefix = match.group(0)
                description = message[len(prefix):].strip()
                return self._clean_commit_message(f"{commit_type}: {description}")
        
        # If message contains a scope-like pattern, try to extract it
        scope_match = re.search(r'\(([\w-]+)\)', message)
        scope = f"({scope_match.group(1)})" if scope_match else ""
        
        # Try to infer type from message content if no valid type is found
        if not any(message.lower().startswith(t) for t in valid_types):
            msg_lower = message.lower()
            
            # Prioritized pattern matching for commit types
            if re.search(r'\b(security|vulnerab|auth|password|encrypt|hash)\b', msg_lower):
                inferred_type = "fix"
                if not scope:
                    scope = "(security)"
            elif re.search(r'\b(fix|bug|issue|error|crash|resolve|problem|incorrect|invalid|wrong)\b', msg_lower):
                inferred_type = "fix"
            elif re.search(r'\b(add|new|feature|implement|support|introduce|enable)\b', msg_lower):
                inferred_type = "feat"
            elif re.search(r'\b(doc|readme|comment|explain|clarify|document)\b', msg_lower):
                inferred_type = "docs"
            elif re.search(r'\b(refactor|clean|reorganize|simplify|restructure|rewrite)\b', msg_lower):
                inferred_type = "refactor"
            elif re.search(r'\b(test|spec|assert|verify|validation)\b', msg_lower):
                inferred_type = "test"
            elif re.search(r'\b(style|format|indent|lint|prettier|eslint)\b', msg_lower):
                inferred_type = "style"
            elif re.search(r'\b(perf|performance|optimize|speed|fast|slow|memory)\b', msg_lower):
                inferred_type = "perf"
            elif re.search(r'\b(dependency|upgrade|update|package|lib|version)\b', msg_lower):
                inferred_type = "chore"
                if not scope:
                    scope = "(deps)"
            elif re.search(r'\b(build|compile|webpack|babel|ci|cd|pipeline|github|jenkins)\b', msg_lower):
                inferred_type = "build"
            else:
                inferred_type = "chore"
            
            # If we have a scope-like part in the text but didn't extract it, remove it from the description
            if scope and scope_match:
                message = message[:scope_match.start()] + message[scope_match.end():]
            
            # Create formatted message with inferred type and scope
            message = f"{inferred_type}{scope}: {message.strip()}"
        
        # Final cleanup
        return self._clean_commit_message(message)
    
    def _clean_commit_message(self, message: str) -> str:
        """Clean up a commit message to ensure it follows the format rules"""
        # Extract type, scope (if any), and description
        match = re.match(r'^(\w+)(\([\w-]+\))?:\s*(.+)$', message)
        if match:
            commit_type = match.group(1).lower()
            scope = match.group(2) or ""
            description = match.group(3).strip()
            
            # Ensure description starts with lowercase
            if description and description[0].isupper():
                description = description[0].lower() + description[1:]
            
            # Remove period at the end
            description = description.rstrip('.')
            
            # Ensure description is concise
            if len(description) > 72:
                description = description[:69] + "..."
            
            return f"{commit_type}{scope}: {description}"
        
        # If pattern doesn't match, return original but strip trailing period
        return message.rstrip('.')
    
    def _simple_commit(self) -> Dict[str, Any]:
        """Perform a simple commit with generated message"""
        # First generate the message
        msg_result = self._generate_commit_message()
        if not msg_result["success"]:
            return msg_result
        
        # Sanitize the commit message to prevent shell injection and escape quotes
        # Remove newlines and escape quotes
        sanitized_message = msg_result["output"].strip().replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
        
        # Then commit with the sanitized message
        commit_result = self._safe_git_command(f'commit -m "{sanitized_message}"')
        return {
            "success": commit_result["success"],
            "output": commit_result["output"],
            "type": "commit",
            "message": msg_result["output"]
        }
    
    def _git_status(self) -> Dict[str, Any]:
        """Get git status"""
        return self._safe_git_command("status")
    
    def _git_diff(self) -> Dict[str, Any]:
        """Get git diff"""
        return self._safe_git_command("diff")
    
    def _git_add(self, request: str) -> Dict[str, Any]:
        """Handle git add operations"""
        if any(word in request.lower() for word in ['all', 'tudo', 'todos', '.']):
            return self._safe_git_command("add -A")
        else:
            # Extract file pattern from request
            # Simple implementation - can be enhanced
            return self._safe_git_command("add .")
    
    def _execute_git_command(self, request: str) -> Dict[str, Any]:
        """Execute arbitrary git command extracted from request"""
        # Extract git command from request
        match = re.search(r'git\s+(.+)', request, re.IGNORECASE)
        if match:
            cmd = match.group(1).strip()
            return self._safe_git_command(cmd)
        else:
            return {
                "success": False,
                "output": "Could not extract git command from request",
                "type": "git_command"
            } 