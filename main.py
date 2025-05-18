import sys
import argparse
import shlex
import re
from agent_core import build_agent
from tools import run_terminal, git_status as git, commit_staged

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
    parts = shlex.split(text.strip(), posix=True)
    return bool(parts) and parts[0] in COMMON_TERMINAL_COMMANDS

def run_once(agent, prompt: str, no_direct: bool):
    if is_terminal_command(prompt) and not no_direct:
        print(f"Executando comando diretamente: {prompt}")
        return run_terminal(prompt)
    
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
    
    try:
        return agent.invoke(prompt)["output"]
    except Exception as e:
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
