#!/usr/bin/env python3
# commit_generator.py - Standalone commit message generator

import os
import subprocess
import shlex
import argparse
import sys

def safe_git_command(cmd: str) -> str:
    """Safely run a git command and return its output"""
    try:
        result = subprocess.run(
            ["git"] + shlex.split(cmd),
            text=True,
            capture_output=True,
            cwd=os.getcwd()
        )
        if result.returncode:
            return result.stderr.strip() or "command failed"
        return result.stdout.strip() or ""
    except Exception as e:
        return f"Error running git command: {str(e)}"

def generate_commit_message(diff_summary: str = "") -> str:
    """Generate a semantic commit message based on the git diff"""
    # Get diff if none provided
    if not diff_summary:
        diff_summary = safe_git_command("diff --stat")
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
    elif any(f.endswith(('.py', '.js', '.tsx', '.jsx', '.ts')) for f in files):
        commit_type = "feat"
    elif any(f.endswith(('.test.js', '.test.ts', '.test.py', '.spec.js', '.spec.ts', '_test.py')) for f in files):
        commit_type = "test"
    elif any('fix' in f.lower() for f in files):
        commit_type = "fix"
    
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

def main():
    parser = argparse.ArgumentParser(description="Generate a semantic commit message based on git diff")
    parser.add_argument("-d", "--diff", help="Use a specific diff summary instead of current git diff")
    parser.add_argument("-c", "--commit", action="store_true", help="Automatically commit with the generated message")
    args = parser.parse_args()

    diff_summary = args.diff or ""
    message = generate_commit_message(diff_summary)
    
    print(message)
    
    if args.commit:
        if message == "No changes detected to generate a commit message":
            print("No changes to commit")
            return 1
        
        # Check if any changes are staged
        staged = safe_git_command("diff --staged --name-only")
        if not staged:
            # If nothing is staged, add all changes
            print("No changes staged, staging all changes...")
            safe_git_command("add .")
            
        # Commit with the generated message
        result = safe_git_command(f"commit -m '{message}'")
        print(result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 