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
    
    # Special case for commit related commands
    commit_patterns = ["commit", "commitar", "comitar", "comite", "commite"]
    if any(pattern in prompt.lower() for pattern in commit_patterns):
        # Handle different commit scenarios
        staged_patterns = ["staged", "stage", "preparado", "adicionado"]
        all_patterns = ["all", "todos", "tudo", "todas"]
        message_patterns = ["message", "mensagem", "descri"]
        
        if any(pattern in prompt.lower() for pattern in staged_patterns):
            print("Executando commit_staged diretamente")
            return commit_staged()
        elif any(pattern in prompt.lower() for pattern in all_patterns + message_patterns):
            # Try to extract message from the prompt
            message = ""
            for pattern in ["message", "mensagem"]:
                if pattern in prompt.lower():
                    parts = prompt.lower().split(pattern, 1)
                    if len(parts) > 1:
                        message = parts[1].strip()
                        break
            
            if not message or len(message) < 5:
                message = "chore: auto commit with descriptive message"
            
            print(f"Executando commit com mensagem: {message}")
            git("add -A")  # Stage all changes
            return git(f'commit -m "{message}"')
        else:
            # Default to commit staged
            print("Executando commit_staged diretamente")
            return commit_staged()
        
    try:
        return agent.invoke(prompt)["output"]
    except Exception as e:
        return f"Erro: {e}"

def explain(user_input: str) -> str:
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
