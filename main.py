import sys
import argparse
import shlex
import re
import glob
from agent_core import build_agent, process_ask_mode
from llm_backend import get_llm
from tools import run_terminal, git_status as git, commit_staged, edit_file, remove_file, create_file, commit_auto
from langchain_core.messages import HumanMessage, SystemMessage
import time

SHARED_LLM = None  # will be set in main()

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

def generate_and_create_file(prompt: str):
    """
    General-purpose function that:
    1. Extracts or infers filename and language
    2. Generates code using LLM to implement the described functionality
    3. Creates the file with the generated code (stripping chain-of-thought)
    """
    # 1) Try to extract an explicit filename with extension
    filename_match = re.search(r'(?:cri[ea]r?|crie|create|make)\s+(?:um\s+)?(?:arquivo|file)?\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt, re.IGNORECASE)
    filename = None
    if filename_match:
        filename = filename_match.group(1).strip()
    
    # If no explicit filename, infer from language keyword
    if not filename:
        # Infer extension from common language words
        lang_ext_map = {
            'javascript': 'js', 'js': 'js', 'typescript': 'ts', 'python': 'py', 'java': 'java', 'c#': 'cs', 'csharp':'cs',
            'c++':'cpp', 'cpp':'cpp', 'c ': 'c', 'go':'go', 'ruby':'rb', 'php':'php', 'swift':'swift',
            'kotlin':'kt', 'rust':'rs', 'bash':'sh', 'shell':'sh'
        }
        for key, ext in lang_ext_map.items():
            if key in prompt.lower():
                filename = f"generated_{int(time.time())}.{ext}"
                break
    if not filename:
        return "Não consegui inferir um nome de arquivo ou linguagem. Por favor, especifique explicitamente o nome do arquivo (ex: meu_arquivo.js)."
    
    extension = filename.split('.')[-1].lower()
    # Map extension to language for prompting
    lang_map = {
        'js': 'JavaScript', 'ts':'TypeScript', 'py':'Python', 'java':'Java', 'cs':'C#', 'cpp':'C++', 'c':'C', 'go':'Go',
        'rb':'Ruby','php':'PHP','swift':'Swift','kt':'Kotlin','rs':'Rust','sh':'Bash','html':'HTML','css':'CSS','sql':'SQL',
        'md':'Markdown','json':'JSON','xml':'XML','yaml':'YAML','yml':'YAML'
    }
    language = lang_map.get(extension, extension.upper())
    
    # 2) Extract functionality description (everything after filename or keyword)
    after_file = prompt
    if filename_match:
        after_file = prompt[filename_match.end():]
    functionality = after_file.strip()
    if not functionality:
        functionality = f"código de exemplo em {language}"
    
    # 3) Generate code with LLM
    llm = SHARED_LLM or get_llm()
    messages = [
        SystemMessage(content="Você é um gerador de código profissional. Gere apenas código, sem explicações."),
        HumanMessage(content=f"Crie código {language} para: {functionality}\nApenas código, sem comentários fora do código.")
    ]
    try:
        code_raw = llm.invoke(messages).content.strip()
    except Exception as e:
        return f"Falha ao obter código do modelo: {e}"
    
    # 4) Strip chain-of-thought or markdown fences
    code = re.sub(r'^```[\w+]*\n', '', code_raw)
    code = re.sub(r'```$', '', code)
    if not code:
        return "Modelo não retornou código válido. Tente reformular o pedido."
    
    # 5) Write file
    result = create_file(f"{filename}|{code}")
    return f"Arquivo {filename} criado com sucesso.\n{result}"

def run_once(agent, prompt: str, no_direct: bool, typo_depth: int = 0):
    # Process file creation requests with more general pattern
    file_creation_pattern = r'(?:cri[ea]r?|crie|create).*?(?:arquivo|file).*?(?:\.[a-z0-9]{1,4}\b|javascript|typescript|python|java|c\+\+|c#|c\b|go\b|ruby|php|swift|kotlin|rust|bash|shell)'
    if re.search(file_creation_pattern, prompt.lower(), re.IGNORECASE):
        print("Detectado pedido de criação de arquivo com código. Gerando código diretamente...")
        return generate_and_create_file(prompt)
    
    # Keep other direct fallbacks for simple commands
    if "tools.py" in prompt.lower() and ("conteudo" in prompt.lower() or "conteúdo" in prompt.lower() or "qual" in prompt.lower()):
        print("Usando solução direta para mostrar conteúdo de tools.py")
        return run_terminal("cat tools.py")
        
    if is_terminal_command(prompt) and not no_direct:
        print(f"Executando comando diretamente: {prompt}")
        return run_terminal(prompt)
        
    # Tratamento direto para comandos comuns
    prompt_lower = prompt.lower()
    
    # Detect combined request to add changes and commit WITH descriptive message before other handlers
    if any(w in prompt_lower for w in ["adicion", "adicione", "add"]) and any(w in prompt_lower for w in ["commit", "commite", "commitar", "comite"]):
        descriptive_req = any(w in prompt_lower for w in ["descrit", "contexto", "diff", "analise", "análise"])
        if descriptive_req:
            print("Fluxo rápido: adicionando e commitando com mensagem descritiva gerada automaticamente")
            return commit_auto(stage_all=True)
        else:
            print("Fluxo rápido: adicionando todas as mudanças e commitando com mensagem padrão")
            git("add -A")
            return commit_staged()

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
    
    # Tratamento simplificado para comandos de commit
    commit_words = ["commit", "commitar", "comitar", "comite", "commite"]
    
    if any(word in prompt.lower() for word in commit_words):
        # Check if user also asked to add/stage changes in same request
        add_requested = any(word in prompt_lower for word in ["add", "adicion", "adicione", "adicionar"])
        descriptive_requested = any(word in prompt_lower for word in ["descritiva", "descrev", "contexto", "diff", "analise", "análise"])

        if add_requested and descriptive_requested:
            print("Adicionando todas as mudanças e commiting com mensagem gerada automaticamente (descritiva)")
            return commit_auto(stage_all=True)
        elif add_requested:
            print("Adicionando todas as mudanças e commiting com mensagem padrão")
            git("add -A")
            return commit_staged()
        elif descriptive_requested:
            print("Commitando arquivos staged com mensagem gerada automaticamente (descritiva)")
            return commit_auto(stage_all=False)
        elif "staged" in prompt_lower:
            print("Executando commit_staged apenas em arquivos já staged")
            return commit_staged()

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
    
    # Tratamento para comandos com erros de digitação comuns
    # MAX_TYPO_DEPTH = 5
    # if typo_depth >= MAX_TYPO_DEPTH:
    #     print(f"Aviso: Profundidade máxima de correção de typos ({MAX_TYPO_DEPTH}) atingida. Processando como está.")
    # else:
    #     typo_patterns = [
    #         (r'\bremo+va\b', 'remover'),
    #         (r'\badicion[ae]\b', 'adicionar'),
    #         (r'\bmodifi[ck]\b', 'modificar'),
    #         (r'\bcommi?ta\b', 'commit'),
    #     ]
    #     
    #     for pattern, replacement in typo_patterns:
    #         if re.search(pattern, prompt_lower):
    #             corrected = re.sub(pattern, replacement, prompt_lower)
    #             if corrected == prompt_lower:
    #                 break  # evita recursão infinita se o padrão corresponder à sua própria saída
    #             print(f"Corrigindo comando: {corrected}")
    #             return run_once(agent, corrected, no_direct, typo_depth + 1) # Incremented typo_depth
    
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
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--no-direct", "-n", action="store_true")
    parser.add_argument("--mode", "-m", choices=["agent", "ask", "free"], default="agent", 
                        help="Modes: agent (tools), ask (conversação sem execução) ou free (LLM livre sem ferramentas)")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    interactive = args.interactive or not args.query

    print("Building agents (shared model)...")
    shared_llm = get_llm()
    global SHARED_LLM
    SHARED_LLM = shared_llm
    agent_mode = build_agent(agent_mode=True, shared_llm=shared_llm)
    raw_llm = build_agent(agent_mode=False, shared_llm=shared_llm)  # retorna apenas o llm
    print("Agents ready!")

    if not interactive:
        query = " ".join(args.query)
        if args.mode == "agent":
            print(run_once(agent_mode, query, args.no_direct))
        elif args.mode == "ask":
            print(process_ask_mode(raw_llm, query))
        else:  # free
            response = raw_llm.invoke([SystemMessage(content="Você é um assistente e deve responder apenas texto."), HumanMessage(content=query)])
            print(response.content)
        return

    # Create session state for interactive mode
    session = {
        "mode": args.mode,
        "conversation_history": [],
        "last_commands": []
    }
    
    print(f"=== Interactive Git-Terminal Assistant === [Mode: {session['mode'].upper()}]")
    print("Tip: Type 'mode agent' to execute commands or 'mode ask' for conversation")
    
    while True:
        try:
            text = input(f"\n[{session['mode'].upper()}] > ")
            if text.strip().lower() in {"exit", "quit", "q"}:
                break
            if not text.strip():
                continue
                
            # Handle mode switching
            if text.lower().startswith("mode "):
                new_mode = text.lower().split(" ")[1].strip()
                if new_mode in ["agent", "ask", "free"]:
                    session["mode"] = new_mode
                    print(f"Switched to {new_mode.upper()} mode")
                    continue
                else:
                    print(f"Unknown mode: {new_mode}. Available modes: agent, ask, free")
                    continue
                    
            # Process based on current mode
            if session["mode"] == "agent":
                if "commit" in text.lower():
                    if input(f"⚠️ {explain(text)}\nContinuar? (s/N): ").lower() != "s":
                        print("Cancelado.")
                        continue
                        
                result = run_once(agent_mode, text, args.no_direct)
                print(result)
                
                # Store last command for context
                session["last_commands"].append({"command": text, "result": result})
                if len(session["last_commands"]) > 5:  # Keep last 5 commands
                    session["last_commands"] = session["last_commands"][-5:]
                    
            elif session["mode"] == "ask":
                # Add conversation context
                context = ""
                if session["conversation_history"] or session["last_commands"]:
                    context = "Based on our conversation and recent commands:\n"
                    # Add last 3 conversation turns
                    if session["conversation_history"]:
                        for turn in session["conversation_history"][-3:]:
                            context += f"- You asked: {turn['question']}\n"
                            context += f"- I answered: {turn['answer'][:100]}...\n"
                    # Add last 2 commands
                    if session["last_commands"]:
                        context += "Recent commands:\n"
                        for cmd in session["last_commands"][-2:]:
                            context += f"- {cmd['command']}\n"
                
                # Process conversational query
                result = process_ask_mode(raw_llm, text, context, session["conversation_history"])
                print(result)
                
                # Store in conversation history
                session["conversation_history"].append({"question": text, "answer": result})
                if len(session["conversation_history"]) > 10:  # Keep last 10 exchanges
                    session["conversation_history"] = session["conversation_history"][-10:]
                
                # Auto-switch detection
                command_patterns = [r'^git\s+\w+', r'^ls\s+', r'^rm\s+', r'^cd\s+', r'^cat\s+']
                if any(re.search(pattern, text.strip()) for pattern in command_patterns):
                    if input("Este parece ser um comando. Mudar para modo Agent? (s/N): ").lower() == "s":
                        session["mode"] = "agent"
                        print("Mudando para AGENT mode e executando comando...")
                        result = run_once(agent_mode, text, args.no_direct)
                        print(result)
                        session["last_commands"].append({"command": text, "result": result})
                
            else:  # free mode
                response = raw_llm.invoke([SystemMessage(content="Você é um assistente e deve responder apenas texto."), HumanMessage(content=text)])
                print(response.content)
                # store conversation simple history
                session["conversation_history"].append({"question": text, "answer": response.content})
                if len(session["conversation_history"]) > 10:
                    session["conversation_history"] = session["conversation_history"][-10:]
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
