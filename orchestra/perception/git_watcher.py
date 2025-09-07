"""
Git repository watcher for monitoring repository state changes and generating perception events.
"""

import os
import subprocess
import threading
import time
from typing import Dict, List, Callable, Optional, Set, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class GitEvent:
    """Represents a git repository event."""
    event_type: str  # status_changed, branch_changed, commit_added, remote_updated, conflict_detected
    repository_path: str
    timestamp: datetime
    details: Dict[str, any]  # Event-specific details


class GitStatus(NamedTuple):
    """Git repository status snapshot."""
    branch: str
    staged_files: List[str]
    modified_files: List[str]
    untracked_files: List[str]
    ahead: int
    behind: int
    has_conflicts: bool
    last_commit_hash: str
    last_commit_message: str


class GitWatcher:
    """Git repository watcher for monitoring repository state changes."""
    
    def __init__(self, 
                 repository_path: str,
                 callback: Callable[[GitEvent], None],
                 check_interval: float = None):
        """
        Initialize Git watcher.
        
        Args:
            repository_path: Path to git repository root
            callback: Function to call when git events occur
            check_interval: How often to check git status (seconds)
        """
        self.repository_path = os.path.abspath(repository_path)
        self.callback = callback
        self.check_interval = check_interval or float(os.getenv('GTA_GIT_CHECK_INTERVAL', '10.0'))
        
        self.running = False
        self.monitor_thread = None
        self.last_status = None
        self.last_check = None
        
        # Validate git repository
        if not self._is_git_repository():
            raise ValueError(f"Not a git repository: {self.repository_path}")
    
    def _is_git_repository(self) -> bool:
        """Check if the path is a git repository."""
        git_dir = os.path.join(self.repository_path, '.git')
        return os.path.exists(git_dir)
    
    def _run_git_command(self, command: List[str]) -> Optional[str]:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Git command failed: git {' '.join(command)}")
                print(f"Error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Git command timed out: git {' '.join(command)}")
            return None
        except Exception as e:
            print(f"Error running git command: {e}")
            return None
    
    def _get_current_status(self) -> Optional[GitStatus]:
        """Get current git repository status."""
        try:
            # Get current branch
            branch_output = self._run_git_command(['branch', '--show-current'])
            if branch_output is None:
                return None
            current_branch = branch_output or 'HEAD'
            
            # Get status
            status_output = self._run_git_command(['status', '--porcelain'])
            if status_output is None:
                return None
            
            staged_files = []
            modified_files = []
            untracked_files = []
            has_conflicts = False
            
            for line in status_output.split('\n'):
                if not line.strip():
                    continue
                    
                status_code = line[:2]
                filename = line[3:]
                
                # Check for conflicts
                if 'UU' in status_code or 'AA' in status_code or 'DD' in status_code:
                    has_conflicts = True
                
                # Parse file status
                if status_code[0] in ['A', 'M', 'D', 'R', 'C']:
                    staged_files.append(filename)
                if status_code[1] in ['M', 'D']:
                    modified_files.append(filename)
                if status_code.startswith('??'):
                    untracked_files.append(filename)
            
            # Get ahead/behind status
            ahead, behind = 0, 0
            ahead_behind_output = self._run_git_command(['rev-list', '--count', '--left-right', f'{current_branch}...origin/{current_branch}'])
            if ahead_behind_output:
                parts = ahead_behind_output.split('\t')
                if len(parts) == 2:
                    ahead = int(parts[0]) if parts[0].isdigit() else 0
                    behind = int(parts[1]) if parts[1].isdigit() else 0
            
            # Get last commit info
            last_commit_hash = self._run_git_command(['rev-parse', 'HEAD']) or ''
            last_commit_message = self._run_git_command(['log', '-1', '--pretty=%s']) or ''
            
            return GitStatus(
                branch=current_branch,
                staged_files=staged_files,
                modified_files=modified_files,
                untracked_files=untracked_files,
                ahead=ahead,
                behind=behind,
                has_conflicts=has_conflicts,
                last_commit_hash=last_commit_hash[:8] if last_commit_hash else '',
                last_commit_message=last_commit_message
            )
            
        except Exception as e:
            print(f"Error getting git status: {e}")
            return None
    
    def _detect_changes(self, old_status: GitStatus, new_status: GitStatus) -> List[GitEvent]:
        """Detect changes between two git status snapshots."""
        events = []
        timestamp = datetime.now()
        
        # Branch change
        if old_status.branch != new_status.branch:
            events.append(GitEvent(
                event_type='branch_changed',
                repository_path=self.repository_path,
                timestamp=timestamp,
                details={
                    'old_branch': old_status.branch,
                    'new_branch': new_status.branch
                }
            ))
        
        # New commit
        if old_status.last_commit_hash != new_status.last_commit_hash:
            events.append(GitEvent(
                event_type='commit_added',
                repository_path=self.repository_path,
                timestamp=timestamp,
                details={
                    'old_commit': old_status.last_commit_hash,
                    'new_commit': new_status.last_commit_hash,
                    'commit_message': new_status.last_commit_message
                }
            ))
        
        # Remote changes (ahead/behind)
        if (old_status.ahead != new_status.ahead or 
            old_status.behind != new_status.behind):
            events.append(GitEvent(
                event_type='remote_updated',
                repository_path=self.repository_path,
                timestamp=timestamp,
                details={
                    'old_ahead': old_status.ahead,
                    'new_ahead': new_status.ahead,
                    'old_behind': old_status.behind,
                    'new_behind': new_status.behind
                }
            ))
        
        # Working directory changes
        old_total_changes = len(old_status.staged_files) + len(old_status.modified_files) + len(old_status.untracked_files)
        new_total_changes = len(new_status.staged_files) + len(new_status.modified_files) + len(new_status.untracked_files)
        
        if old_total_changes != new_total_changes:
            events.append(GitEvent(
                event_type='status_changed',
                repository_path=self.repository_path,
                timestamp=timestamp,
                details={
                    'staged_files': new_status.staged_files,
                    'modified_files': new_status.modified_files,
                    'untracked_files': new_status.untracked_files,
                    'total_changes': new_total_changes
                }
            ))
        
        # Conflict detection
        if not old_status.has_conflicts and new_status.has_conflicts:
            events.append(GitEvent(
                event_type='conflict_detected',
                repository_path=self.repository_path,
                timestamp=timestamp,
                details={
                    'branch': new_status.branch,
                    'conflicted_files': [f for f in new_status.staged_files + new_status.modified_files 
                                       if 'conflict' in f.lower()]
                }
            ))
        
        return events
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        print(f"🔍 Starting Git watcher for {self.repository_path}")
        
        # Get initial status
        self.last_status = self._get_current_status()
        if self.last_status is None:
            print("⚠️ Could not get initial git status")
            return
        
        self.last_check = time.time()
        
        while self.running:
            try:
                time.sleep(self.check_interval)
                
                if not self.running:
                    break
                
                current_status = self._get_current_status()
                if current_status is None:
                    continue
                
                # Detect changes
                if self.last_status:
                    events = self._detect_changes(self.last_status, current_status)
                    
                    # Send events to callback
                    for event in events:
                        try:
                            self.callback(event)
                        except Exception as e:
                            print(f"Error in Git event callback: {e}")
                
                self.last_status = current_status
                self.last_check = time.time()
                
            except Exception as e:
                print(f"Error in git monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def start(self):
        """Start monitoring git repository."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring git repository."""
        if not self.running:
            return
        
        print("🛑 Stopping Git watcher")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_current_status(self) -> Optional[GitStatus]:
        """Get current git status (public interface)."""
        return self._get_current_status()
    
    def get_watch_info(self) -> Dict[str, any]:
        """Get information about current watching state."""
        return {
            "running": self.running,
            "repository_path": self.repository_path,
            "check_interval": self.check_interval,
            "last_check": self.last_check,
            "current_status": self.last_status._asdict() if self.last_status else None
        }
    
    def force_check(self) -> List[GitEvent]:
        """Force a status check and return any detected events."""
        if not self.last_status:
            self.last_status = self._get_current_status()
            return []
        
        current_status = self._get_current_status()
        if current_status is None:
            return []
        
        events = self._detect_changes(self.last_status, current_status)
        self.last_status = current_status
        self.last_check = time.time()
        
        return events


def create_project_git_watcher(project_root: str, callback: Callable[[GitEvent], None]) -> Optional[GitWatcher]:
    """Create a GitWatcher for a project if it's a git repository."""
    
    # Look for git repository
    git_root = find_git_root(project_root)
    if not git_root:
        return None
    
    return GitWatcher(
        repository_path=git_root,
        callback=callback,
        check_interval=float(os.getenv('GTA_GIT_CHECK_INTERVAL', '15.0'))
    )


def find_git_root(path: str) -> Optional[str]:
    """Find the git repository root starting from the given path."""
    current_path = os.path.abspath(path)
    
    while current_path != os.path.dirname(current_path):  # Not at filesystem root
        git_dir = os.path.join(current_path, '.git')
        if os.path.exists(git_dir):
            return current_path
        current_path = os.path.dirname(current_path)
    
    return None


def get_repository_info(repo_path: str) -> Dict[str, any]:
    """Get detailed information about a git repository."""
    watcher = GitWatcher(repo_path, lambda e: None)
    status = watcher.get_current_status()
    
    if not status:
        return {"error": "Could not get repository status"}
    
    # Get additional repository info
    def run_git(cmd):
        try:
            result = subprocess.run(
                ['git'] + cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    # Get remote info
    remote_url = run_git(['remote', 'get-url', 'origin'])
    remote_branches = run_git(['branch', '-r'])
    
    # Get recent commits
    recent_commits = run_git(['log', '--oneline', '-10'])
    
    return {
        "status": status._asdict(),
        "remote_url": remote_url,
        "remote_branches": remote_branches.split('\n') if remote_branches else [],
        "recent_commits": recent_commits.split('\n') if recent_commits else [],
        "repository_path": repo_path
    }
