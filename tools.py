import os, subprocess, shlex, re
from pathlib import Path
from pydantic import BaseModel
from langchain.agents import Tool
from commit_generator import generate_commit_message

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

def create_file(file_info: str) -> str:
    """Cria um arquivo com o conteúdo especificado. Formato: 'caminho|conteúdo'"""
    try:
        parts = file_info.split('|', 1)
        if len(parts) != 2:
            return "Formato incorreto. Use: 'caminho|conteúdo'"
        
        path, content = parts
        Path(path.strip()).expanduser().write_text(content)
        return f"Arquivo {path.strip()} criado com sucesso."
    except Exception as e:
        return f"Erro ao criar arquivo: {str(e)}"

def edit_file(file_info: str) -> str:
    """Edita um arquivo existente. Formato: 'caminho|conteúdo'"""
    return create_file(file_info)

def remove_file(filepath: str) -> str:
    """Remove um arquivo especificado."""
    try:
        path = Path(filepath.strip()).expanduser()
        if not path.exists():
            return f"Arquivo {filepath} não encontrado."
        path.unlink()
        return f"Arquivo {filepath} removido com sucesso."
    except Exception as e:
        return f"Erro ao remover arquivo: {str(e)}"

def commit_auto(stage_all: bool = False) -> str:
    """Stage files (optionally all) and commit with an auto-generated descriptive message."""
    try:
        if stage_all:
            _safe_run(["git", "add", "-A"])
        # Ensure there is something staged
        staged_files = git_status("diff --cached --name-only")
        if not staged_files:
            return "Nenhum arquivo staged para commit."
        # Build diff summary for staged changes only
        diff_summary = _safe_run(["git", "diff", "--cached", "--stat"])
        message = generate_commit_message(diff_summary)
        # Perform commit
        return _safe_run(["git", "commit", "-m", message])
    except Exception as e:
        return f"Erro ao commitar: {str(e)}"

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
CreateFileTool = Tool("create_file", create_file, 
                     "Use para criar um novo arquivo. O input DEVE ser 'caminho/do/arquivo.txt|conteúdo textual do arquivo'. Se o prompt do usuário descrever o conteúdo (ex: 'crie um arquivo com uma função que faz X'), você DEVE PRIMEIRO GERAR O CONTEÚDO COMPLETO e então usar esta ferramenta com o nome do arquivo e o conteúdo gerado. Não use esta ferramenta sem conteúdo explícito.")
EditFileTool = Tool("edit_file", edit_file,
                    "Edita arquivo existente; formato: 'arquivo.txt|novo conteúdo'")
RemoveFileTool = Tool("remove_file", remove_file,
                      "Remove um arquivo; arg: path do arquivo")
CommitAutoTool = Tool("commit_auto", lambda stage_all=False: commit_auto(stage_all),
                       description="Adiciona (opcionalmente) todas as mudanças e commita com mensagem gerada automaticamente")

ALL_TOOLS = [TerminalTool, GitStatusTool, CommitStagedTool, FileRead, FileWrite, 
             CreateFileTool, EditFileTool, RemoveFileTool, CommitAutoTool]
