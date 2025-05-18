import os, subprocess, shlex, re
from pathlib import Path
from pydantic import BaseModel
from langchain.agents import Tool

def _safe_run(cmd: list[str]) -> str:
    if re.search(r"rm\s+-[rf].*\/", " ".join(cmd)):
        return "⚠️ BLOCKED: comando perigoso."
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    return p.stdout.strip() or p.stderr.strip() or "✓ done"

def run_terminal(command: str) -> str:
    return _safe_run(["/bin/zsh", "-c", command])

def git_status(cmd: str) -> str:
    return _safe_run(["git"] + shlex.split(cmd))

def commit_staged() -> str:
    staged = git_status("diff --cached --name-only")
    if not staged:
        return "Nenhum arquivo staged para commit."
    return git_status('commit -m "chore: commit staged changes"')

class _NoArgs(BaseModel): pass

TerminalTool     = Tool("terminal",       run_terminal,  "Executa comando de shell")
GitStatusTool    = Tool("git_status",     git_status,    "git <subcomando> somente leitura")
CommitStagedTool = Tool("commit_staged",  lambda *args, **kwargs: commit_staged(), 
                        description="Commita somente arquivos já staged")
FileRead  = Tool("read_file",  lambda p: Path(p).expanduser().read_text(),          
                 "Lê arquivo texto")
FileWrite = Tool("write_file", lambda s: Path(s.split('|',1)[0]).expanduser()
                                          .write_text(s.split('|',1)[1]) or "ok",
                 "Cria ou sobrescreve arquivo; arg: path|conteúdo")

ALL_TOOLS = [TerminalTool, GitStatusTool, CommitStagedTool, FileRead, FileWrite]
