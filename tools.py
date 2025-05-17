# tools.py
import os, subprocess, shlex
from pathlib import Path
from langchain.agents import Tool
import re
import tempfile

# ---------- low-level helpers ----------
def _safe_run(cmd: list[str]) -> str:
    # Safety check for dangerous commands
    cmd_str = " ".join(cmd)
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r"rm\s+(-r|-rf|--recursive)\s+/",  # Remove root or important directories
        r"dd\s+.*(of=|if=)/dev/",          # Disk operations on devices
        r"mkfs",                           # Format filesystems
        r":(){:\|:&};:",                   # Fork bomb
        r">(>?)\s*/dev/(sd[a-z]|nvme|hd)"  # Write to disk devices
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, cmd_str, re.IGNORECASE):
            return "⚠️ BLOCKED: Potentially dangerous command detected. For safety reasons, this command was not executed."
    
    # For rm commands with wildcards, run in a temporary directory to avoid accidents
    if "rm" in cmd_str and ("*" in cmd_str or "-r" in cmd_str or "-rf" in cmd_str):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run in temp dir if not explicitly targeting a specific path
            if not any(os.path.exists(arg) for arg in cmd[1:] if not arg.startswith("-")):
                completed = subprocess.run(
                    cmd, text=True, capture_output=True, cwd=tmpdir
                )
                return f"⚠️ WARNING: 'rm' command with wildcards or recursive flags was run in a safe temporary directory instead of the current directory."
    
    # Standard execution
    completed = subprocess.run(
        cmd, text=True, capture_output=True, cwd=os.getcwd()
    )
    if completed.returncode:
        return completed.stderr.strip() or "command failed"
    return completed.stdout.strip() or "✓ done"

# ---------- Terminal ----------
def run_terminal(command: str) -> str:
    return _safe_run(["/bin/zsh", "-c", command])

TerminalTool = Tool(
    name="terminal",
    func=run_terminal,
    description="Run arbitrary macOS shell commands (pipes, redirs allowed)"
)

# ---------- File Tree Visualizer ----------
def visualize_file_tree(max_level: str = "2") -> str:
    """Display file tree structure using tree command"""
    try:
        # First check if tree is installed
        which_result = subprocess.run(["which", "tree"], capture_output=True, text=True)
        if which_result.returncode != 0:
            return "Tree command not found. Please install it with 'brew install tree' or use ls -R instead."
        
        # Try to parse max_level as integer, default to 2 if not possible
        try:
            level = int(max_level)
            if level < 1:
                level = 2
            elif level > 4:  # Limit depth to avoid excessive output
                level = 4
        except ValueError:
            level = 2
            
        # Run tree command with color and limited level
        result = _safe_run(["tree", "-L", str(level), "-C"])
        return result
    except Exception as e:
        return f"Error generating file tree: {str(e)}"

FileTreeTool = Tool(
    name="file_tree",
    func=visualize_file_tree,
    description="Visualize the file tree structure with optional depth level (default 2, max 4)"
)

# ---------- Git ----------
def git(cmd: str) -> str:
    return _safe_run(["git"] + shlex.split(cmd))

GitTool = Tool(
    name="git",
    func=git,
    description="Run Git sub-commands, e.g. `status`, `add .`, `commit -m msg`, `push`"
)

# ---------- Files ----------
def read_file(path: str) -> str:
    p = Path(path).expanduser()
    return p.read_text(encoding="utf-8")

def write_file(spec: str) -> str:
    path, content = spec.split("|", 1)
    Path(path).expanduser().write_text(content, encoding="utf-8")
    return f"wrote {path}"

def append_file(spec: str) -> str:
    path, content = spec.split("|", 1)
    p = Path(path).expanduser()
    if p.exists():
        current = p.read_text(encoding="utf-8")
        p.write_text(current + content, encoding="utf-8")
        return f"appended to {path}"
    else:
        p.write_text(content, encoding="utf-8")
        return f"created and wrote to {path}"

def list_files(directory: str = ".") -> str:
    if not directory:
        directory = "."
    p = Path(directory).expanduser()
    if not p.exists():
        return f"Directory {directory} does not exist"
    if not p.is_dir():
        return f"{directory} is not a directory"
    
    files = []
    dirs = []
    
    for item in p.iterdir():
        if item.is_dir():
            dirs.append(f"{item.name}/")
        else:
            files.append(item.name)
    
    dirs.sort()
    files.sort()
    
    result = f"Contents of {directory}:\n"
    if dirs:
        result += "Directories:\n  " + "\n  ".join(dirs) + "\n"
    if files:
        result += "Files:\n  " + "\n  ".join(files)
    
    return result

FileRead = Tool(name="read_file", func=read_file, description="Read a text file")
FileWrite = Tool(name="write_file", func=write_file, description="Overwrite or create a file; arg format `path|content`")
FileAppend = Tool(name="append_file", func=append_file, description="Append to a file; arg format `path|content`")
ListFiles = Tool(name="list_files", func=list_files, description="List files in a directory; defaults to current dir")

# ---------- Git Semantic Commit ----------
def generate_commit_message(diff_summary: str = "") -> str:
    """Generate a semantic commit message based on the current git diff"""
    # Get diff if none provided
    if not diff_summary:
        diff_summary = git("diff --stat")
        if not diff_summary:
            return "No changes detected to generate a commit message"
    
    # Analyze what files changed
    files = []
    for line in diff_summary.split('\n'):
        if '|' in line and line.strip():
            filename = line.split('|')[0].strip()
            if filename:
                files.append(filename)
    
    # Determine type of change
    commit_type = "chore"
    if any(f.endswith(('.md', '.txt', 'README')) for f in files):
        commit_type = "docs"
    elif any(f.endswith(('.py', '.js', '.tsx')) for f in files):
        commit_type = "feat"
    
    # Determine scope
    scope = ""
    if files:
        common_prefix = os.path.commonprefix(files)
        if common_prefix and '/' in common_prefix:
            scope = f"({common_prefix.split('/')[0]})"
    
    # Create message
    if files:
        files_str = ", ".join(files[:3])
        if len(files) > 3:
            files_str += f" and {len(files) - 3} more"
        message = f"{commit_type}{scope}: update {files_str}"
    else:
        message = f"{commit_type}: automatic update"
    
    return message

CommitMessageTool = Tool(
    name="generate_commit_message",
    func=generate_commit_message,
    description="Generate a semantic commit message based on the current git diff"
)

# Combine all tools
ALL_TOOLS = [
    TerminalTool,
    GitTool, 
    FileRead, 
    FileWrite, 
    FileAppend,
    ListFiles,
    FileTreeTool,
    CommitMessageTool
] 