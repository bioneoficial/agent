import sys
import argparse
import shlex
import re
import glob
from new_agent import build_agent
from tools import run_terminal, git_status as git, commit_staged, edit_file, remove_file, create_file

COMMON_TERMINAL_COMMANDS = [
    "ls","ll","la","pwd","cd","find","locate","tree","whoami","who","w","id",
    "uname","hostname","uptime","date","cal","cat","less","more","head","tail",
    "touch","mkdir","rmdir","env","echo","ps","top","htop","kill","killall",
    "jobs","bg","fg","ping","ifconfig","ip","netstat","ssh","curl","wget",
    "telnet","nc","clear","history","man","info","help","which","whereis",
    "alias","tar","gzip","gunzip","zip","unzip","bzip2","bunzip2","nano",
    "vim","vi","emacs","pico"
]

def is_terminal_command(text: str) -> bool:
    try:
        parts = shlex.split(text.strip(), posix=True)
        return bool(parts) and parts[0] in COMMON_TERMINAL_COMMANDS
    except:
        return False

def extract_command(prompt: str, command_type: str) -> str:
    """Extrai comandos específicos do prompt do usuário"""
    if command_type == "git":
        match = re.search(r'git\s+([^;|&<>]+)', prompt, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    if command_type == "ls":
        match = re.search(r'(?:ls|listar|exibir)\s+([^;|&<>]*)', prompt, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    if command_type == "cat":
        match = re.search(r'(?:cat|mostrar|exibir)\s+(?:o )?(?:conteúdo )?(?:do )?(?:arquivo )?([^\s;|&<>]+)', prompt, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    if command_type == "rm":
        # Improved pattern matching for files with specific extensions
        ext_match = re.search(r'(?:remov|delet|rm|apag)\w*\s+(?:o[s]? )?(?:arquivo[s]? )?(?:com|que\s+(?:tenha|contenha)[m]?|(?:com\s+)?(?:nome|padrão))\s+(\.[a-zA-Z0-9]+)', prompt, re.IGNORECASE)
        if ext_match:
            return f"extension:{ext_match.group(1).strip()}"
            
        # Standard pattern matching
        pattern_match = re.search(r'(?:remov|delet|rm|apag)\w*\s+(?:o[s]? )?(?:arquivo[s]? )?(?:com|que\s+(?:tenha|contenha)[m]?|(?:com\s+)?(?:nome|padrão))\s+([^\s;|&<>]+)', prompt, re.IGNORECASE)
        if pattern_match:
            return f"pattern:{pattern_match.group(1).strip()}"
            
        # Caso padrão para arquivo específico
        match = re.search(r'(?:remov|delet|rm|apag)\w*\s+(?:o )?(?:arquivo )?([^\s;|&<>]+)', prompt, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    if command_type == "edit":
        # Extrai o nome do arquivo a ser editado
        file_match = re.search(r'edit\w*\s+(?:o )?(?:conte(?:ú|u)do )?(?:do )?(?:arquivo )?([^\s;|&<>]+)', prompt, re.IGNORECASE)
        if not file_match:
            return ""
        
        # Extrai o novo conteúdo após "para" ou ":"
        content_match = re.search(r'(?:para|com|:|"|\').*?([^"\'].*?)(?:$|"|\')', prompt, re.IGNORECASE | re.DOTALL)
        if not content_match:
            return ""
            
        file_path = file_match.group(1).strip()
        content = content_match.group(1).strip()
        return f"{file_path}|{content}"
        
    return ""

def run_once(agent, prompt: str, no_direct: bool):
    if is_terminal_command(prompt) and not no_direct:
        print(f"Executando comando diretamente: {prompt}")
        return run_terminal(prompt)
        
    # Tratamento direto para comandos comuns
    prompt_lower = prompt.lower()
    
    # Comando git
    if "git " in prompt_lower or ("adicione" in prompt_lower and "git" in prompt_lower) or "staged" in prompt_lower or "stage" in prompt_lower:
        # Tratamento especial para git restore (unstage)
        restore_patterns = [
            r'git\s+restore\s+.*?staged',
            r'git\s+restore\s+--staged',
            r'(?:tire|remov|delet|limpe|tira)\s+(?:os?\s+)?(?:arquivo[s]?)?(?:\s+d[eo])?(?:\s+)?staged',
            r'(?:tire|remov|delet|limpe|tira)\s+(?:os?\s+)?(?:arquivo[s]?)?\s+(?:d[eo]\s+)?stage',
            r'(?:tire|remov|delet|limpe|tira)\s+(?:os?\s+)?(?:arquivo[s]?)?\s+(?:d[eo]\s+)?staged\s+changes',
            r'unstage',
            r'desfaz(?:er)?\s+(?:os?\s+)?staged',
            r'restore\s+(?:(?:os|d[eo]s)\s+)?(?:arquivo[s]?)?\s+(?:em\s+)?stage',
            r'(?:tire|remov|unstage)\s+(?:os?\s+)?(?:arquivo[s]?)?\s+(?:de\s+)?(stage|staged)'
        ]
        
        # Check each pattern individually and print if matched for debugging
        matched = False
        for pattern in restore_patterns:
            if re.search(pattern, prompt_lower):
                print(f"Padrão de unstage encontrado: {pattern}")
                matched = True
                break
                
        if matched or "staged changes" in prompt_lower:
            print("Executando git restore --staged para remover arquivos do stage")
            return git("restore --staged .")
            
        # Caso especial para adicionar arquivos modificados ao git
        modified_patterns = [
            r'adicione\s+(?:ao\s+)?git\s+(?:os?\s+)?(?:arquivos?\s+)?modific',
            r'adicione\s+(?:todos?\s+)?(?:os\s+)?(?:arquivos?\s+)?modific(?:\w+)?\s+(?:ao|no)\s+git',
            r'git\s+add\s+(?:os\s+)?modific',
            r'adicione\s+(?:as\s+)?(?:mudanças|alterações)',
            r'git\s+adicione\s+(?:tudo|tods|todos)',
            r'adicione\s+(?:todos|tudo)\s+(?:ao|no)\s+git',
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in modified_patterns):
            print("Adicionando todos os arquivos modificados ao git")
            return git("add .")
            
        # Caso especial para adicionar arquivo específico ao git
        pattern = r'adicione\s+(?:ao\s+)?git\s+(?:o\s+)?(?:arquivo\s+)?([^\s;|&<>]+)'
        add_match = re.search(pattern, prompt, re.IGNORECASE)
        if add_match:
            file = add_match.group(1).strip()
            # Verifica se não é uma palavra comum que pode gerar erro
            if file not in ['o', 'os', 'as', 'do', 'dos', 'das', 'no', 'nos', 'nas', 'um', 'uma']:
                print(f"Adicionando {file} ao git")
                return git(f"add {file}")
            else:
                # Se for palavra comum, tentar adicionar todos os arquivos modificados
                print("Adicionando todos os arquivos modificados ao git")
                return git("add .")
            
        # Caso normal de comando git
        git_subcmd = extract_command(prompt, "git")
        if git_subcmd:
            print(f"Executando git {git_subcmd}")
            return git(git_subcmd)
    
    # Comando ls/listar diretório
    if "ls " in prompt_lower or "listar " in prompt_lower or ("mostrar" in prompt_lower and "diretório" in prompt_lower):
        ls_args = extract_command(prompt, "ls") or "-la"
        print(f"Listando diretório: ls {ls_args}")
        return run_terminal(f"ls {ls_args}")
    
    # Comando para mostrar conteúdo de arquivo
    if "cat " in prompt_lower or ("mostrar" in prompt_lower and "conteúdo" in prompt_lower) or ("exibir" in prompt_lower and "arquivo" in prompt_lower):
        file_path = extract_command(prompt, "cat")
        if file_path:
            print(f"Mostrando conteúdo do arquivo: {file_path}")
            return run_terminal(f"cat {file_path}")
            
    # Comando para remover arquivos
    if any(word in prompt_lower for word in ["remov", "delet", "apag", "exclu", "rm "]):
        file_path = extract_command(prompt, "rm")
        if file_path:
            # Handle file extension pattern
            if file_path.startswith("extension:"):
                extension = file_path[10:]  # Remove the prefix "extension:"
                print(f"Removendo arquivos com extensão: {extension}")
                matches = glob.glob(f"*{extension}")
                if not matches:
                    return f"Nenhum arquivo encontrado com a extensão '{extension}'."
                
                results = []
                for file_match in matches:
                    try:
                        result = remove_file(file_match)
                        results.append(result)
                    except Exception as e:
                        results.append(f"Erro ao remover {file_match}: {str(e)}")
                
                return "\n".join(results)
            
            # Tratamento especial para padrões de arquivo
            if file_path.startswith("pattern:"):
                pattern = file_path[8:]  # Remove o prefixo "pattern:"
                print(f"Removendo arquivos com padrão: {pattern}")
                matches = glob.glob(f"*{pattern}*")
                if not matches:
                    return f"Nenhum arquivo encontrado com o padrão '{pattern}'."
                
                results = []
                for file_match in matches:
                    try:
                        result = remove_file(file_match)
                        results.append(result)
                    except Exception as e:
                        results.append(f"Erro ao remover {file_match}: {str(e)}")
                
                return "\n".join(results)
            else:
                print(f"Removendo arquivo: {file_path}")
                return remove_file(file_path)
            
    # Comando para editar conteúdo de um arquivo
    if "edit" in prompt_lower or ("modific" in prompt_lower and "conteúdo" in prompt_lower):
        file_content = extract_command(prompt, "edit")
        if file_content and "|" in file_content:
            file_path, content = file_content.split("|", 1)
            print(f"Editando arquivo {file_path} com novo conteúdo: {content}")
            return edit_file(f"{file_path}|{content}")
    
    # Tratamento especial para comandos de criação de arquivo
    create_file_match = re.search(r'cri(?:ar|e) (?:um )?arquivo (?:chamado )?([^\s]+).*(?:conte[úu]do|com)\s*:?\s*[\'"]?([^\'""]+)[\'"]?', prompt, re.IGNORECASE | re.DOTALL)
    if create_file_match:
        file_path = create_file_match.group(1).strip()
        content = create_file_match.group(2).strip()
        
        if file_path and content:
            try:
                print(f"Criando arquivo {file_path} com conteúdo: {content}")
                result = create_file(f"{file_path}|{content}")
                return result
            except Exception as e:
                print(f"Erro ao criar arquivo: {e}")
    
    # Tratamento simplificado para comandos de commit
    commit_words = ["commit", "commitar", "comitar", "comite", "commite"]
    
    if any(word in prompt.lower() for word in commit_words):
        all_patterns = ["all", "todos", "tudo", "todas", "tds", "everything"]
        commit_all = any(pattern in prompt.lower() for pattern in all_patterns)
        
        if commit_all:
            # Extrai a mensagem de commit, se houver
            message = ""
            for pattern in ["message", "mensagem", "with", "com"]:
                if pattern in prompt.lower():
                    parts = prompt.lower().split(pattern, 1)
                    if len(parts) > 1:
                        message = parts[1].strip()
                        break
            
            if not message or len(message) < 5:
                # Se não houver mensagem específica, usa uma mensagem genérica
                message = "chore: auto commit with all changes"
            
            print(f"Executando commit de todas as alterações com mensagem: {message}")
            git("add -A")  # Stage all changes
            return git(f'commit -m "{message}"')
            
        elif "staged" in prompt.lower() or "stage" in prompt.lower():
            # Apenas commita os arquivos que já estão staged
            print("Executando commit_staged diretamente")
            return commit_staged()
        else:
            # Tenta determinar o que o usuário quer com base no texto
            if "descri" in prompt.lower() or "mensagem" in prompt.lower() or "message" in prompt.lower():
                # Provavelmente quer commitar todos com mensagem
                message = prompt.split("message", 1)[-1].strip() if "message" in prompt.lower() else ""
                if not message or len(message) < 5:
                    message = prompt.split("mensagem", 1)[-1].strip() if "mensagem" in prompt.lower() else ""
                if not message or len(message) < 5:
                    message = "chore: auto commit based on user request"
                
                print(f"Executando commit com mensagem: {message}")
                git("add -A")  # Stage all changes
                return git(f'commit -m "{message}"')
            else:
                # Default: apenas commita os arquivos staged
                print("Executando commit_staged diretamente")
                return commit_staged()
    
    # Tratamento para comandos com erros de digitação comuns
    typo_patterns = [
        (r'remo+va', 'remover'),
        (r'adicion[ae]', 'adicionar'),
        (r'modifi[ck]', 'modificar'),
        (r'commi?ta', 'commit'),
    ]
    
    for pattern, replacement in typo_patterns:
        if re.search(pattern, prompt_lower):
            # Tentamos corrigir o erro e reprocessar
            corrected = re.sub(pattern, replacement, prompt_lower)
            print(f"Corrigindo comando: {corrected}")
            # Tenta executar a versão corrigida
            return run_once(agent, corrected, no_direct)
    
    try:
        return agent.invoke(prompt)["output"]
    except Exception as e:
        # Melhoria no tratamento de erros
        error_msg = str(e)
        if "tools" in error_msg:
            # Trata erros comuns de ferramentas
            if any(word in prompt_lower for word in ["remov", "delet", "apag"]):
                # Tenta extrair um padrão ou nome de arquivo mais específico
                file_pattern_match = re.search(r'(?:remov|delet|rm|apag).*?(?:com|contendo|padrão)\s+([^\s;|&<>]+)', prompt, re.IGNORECASE)
                if file_pattern_match:
                    pattern = file_pattern_match.group(1).strip()
                    if pattern:
                        print(f"Tentando remover arquivos com padrão: {pattern}")
                        matches = glob.glob(f"*{pattern}*")
                        if matches:
                            results = []
                            for file_match in matches:
                                try:
                                    result = remove_file(file_match)
                                    results.append(result)
                                except Exception as e2:
                                    results.append(f"Erro ao remover {file_match}: {str(e2)}")
                            return "\n".join(results)
                        else:
                            return f"Nenhum arquivo encontrado com o padrão '{pattern}'."
            
            if "adicione" in prompt_lower and "git" in prompt_lower:
                # Caso de erro ao tentar adicionar algo ao git
                print("Tentando adicionar todas as mudanças ao git")
                return git("add .")
                
            # Handle git restore/unstage errors
            if any(pattern in prompt_lower for pattern in ["unstage", "restore", "tire", "remov"]) and any(pattern in prompt_lower for pattern in ["stage", "staged"]):
                print("Tentando remover arquivos do stage")
                return git("restore --staged .")
                
        return f"Erro: {e}"

def explain(user_input: str) -> str:
    # Verificar se o comando é para commitar tudo ou apenas arquivos staged
    all_patterns = ["all", "todos", "tudo", "todas", "tds", "everything"]
    commit_all = any(pattern in user_input.lower() for pattern in all_patterns)
    
    if commit_all:
        # Conta tanto arquivos staged quanto não staged
        staged = git("diff --cached --name-only").splitlines()
        unstaged = git("diff --name-only").splitlines()
        all_files = list(set(staged + unstaged))
        if all_files:
            return f"Esta ação irá commitar {len(all_files)} arquivo(s) (incluindo não staged)."
        return "Nenhuma alteração para commitar."
    else:
        # Apenas arquivos staged
        staged = git("diff --cached --name-only").splitlines()
        if staged:
            return f"Esta ação irá commitar {len(staged)} arquivo(s) staged."
        return "Nenhum arquivo staged para commit."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive","-i",action="store_true")
    parser.add_argument("--no-direct","-n",action="store_true")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    interactive = args.interactive or not args.query

    print("Building agent...")
    agent = build_agent()
    print("LangChain agent ready!")

    if not interactive:
        query = " ".join(args.query)
        print(run_once(agent, query, args.no_direct))
        return

    print("=== Interactive Git-Terminal Assistant ===")
    while True:
        try:
            text = input("\n> ")
            if text.strip().lower() in {"exit","quit","q"}:
                break
            if not text.strip():
                continue

            if "commit" in text.lower():
                if input(f"⚠️ {explain(text)}\nContinuar? (s/N): ").lower() != "s":
                    print("Cancelado.")
                    continue

            result = run_once(agent, text, args.no_direct)
            print(result)
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
