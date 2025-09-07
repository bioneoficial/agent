"""
Filesystem watcher for monitoring file changes and generating perception events.
"""

import os
import time
import threading
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import fnmatch

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object  # Use object as base class when watchdog unavailable
    FileSystemEvent = None


@dataclass
class FSEvent:
    """Represents a filesystem event."""
    event_type: str  # created, modified, deleted, moved
    path: str
    is_directory: bool
    timestamp: datetime
    size: Optional[int] = None
    old_path: Optional[str] = None  # for moved events


class FSWatcherHandler(FileSystemEventHandler):
    """Handler for filesystem events from watchdog."""
    
    def __init__(self, callback: Callable[[FSEvent], None], ignore_patterns: List[str]):
        super().__init__()
        self.callback = callback
        self.ignore_patterns = ignore_patterns
        
    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored based on patterns."""
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False
    
    def _create_event(self, event: FileSystemEvent, event_type: str) -> FSEvent:
        """Create FSEvent from watchdog event."""
        size = None
        if not event.is_directory and os.path.exists(event.src_path):
            try:
                size = os.path.getsize(event.src_path)
            except (OSError, IOError):
                pass
                
        return FSEvent(
            event_type=event_type,
            path=event.src_path,
            is_directory=event.is_directory,
            timestamp=datetime.now(),
            size=size,
            old_path=getattr(event, 'dest_path', None)
        )
    
    def on_created(self, event):
        if not self._should_ignore(event.src_path):
            fs_event = self._create_event(event, 'created')
            self.callback(fs_event)
    
    def on_modified(self, event):
        if not self._should_ignore(event.src_path):
            fs_event = self._create_event(event, 'modified')
            self.callback(fs_event)
    
    def on_deleted(self, event):
        if not self._should_ignore(event.src_path):
            fs_event = self._create_event(event, 'deleted')
            self.callback(fs_event)
    
    def on_moved(self, event):
        if not self._should_ignore(event.src_path) and not self._should_ignore(event.dest_path):
            fs_event = self._create_event(event, 'moved')
            fs_event.old_path = event.src_path
            fs_event.path = event.dest_path
            self.callback(fs_event)


class FSWatcher:
    """Filesystem watcher for monitoring directory changes."""
    
    DEFAULT_IGNORE_PATTERNS = [
        # Version control
        '.git/*', '.svn/*', '.hg/*',
        # IDE/Editor files
        '.vscode/*', '.idea/*', '*.swp', '*.swo', '*~',
        # Build artifacts
        '__pycache__/*', '*.pyc', '*.pyo', '.pytest_cache/*',
        'node_modules/*', 'build/*', 'dist/*', '.tox/*',
        # Logs and temp files
        '*.log', '*.tmp', '.DS_Store', 'Thumbs.db',
        # Orchestra system files
        '.orchestra/*',
        # Common ignore patterns
        '*.bak', '*.backup', '*.old'
    ]
    
    def __init__(self, 
                 watch_paths: List[str], 
                 callback: Callable[[FSEvent], None],
                 ignore_patterns: Optional[List[str]] = None,
                 recursive: bool = True):
        """
        Initialize filesystem watcher.
        
        Args:
            watch_paths: Directories to watch
            callback: Function to call when events occur
            ignore_patterns: File patterns to ignore (None for defaults)
            recursive: Whether to watch subdirectories
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog package is required for filesystem watching")
            
        self.watch_paths = [os.path.abspath(path) for path in watch_paths]
        self.callback = callback
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.recursive = recursive
        
        self.observer = Observer()
        self.handler = FSWatcherHandler(self._handle_event, self.ignore_patterns)
        self.running = False
        self.event_buffer = []
        self.last_buffer_flush = time.time()
        self.buffer_lock = threading.Lock()
        
        # Configuration
        self.debounce_interval = float(os.getenv('GTA_FS_DEBOUNCE_INTERVAL', '0.5'))
        self.buffer_flush_interval = float(os.getenv('GTA_FS_BUFFER_FLUSH_INTERVAL', '2.0'))
        
    def _handle_event(self, event: FSEvent):
        """Handle filesystem event with debouncing."""
        with self.buffer_lock:
            self.event_buffer.append(event)
            
            # Check if we should flush buffer
            now = time.time()
            if now - self.last_buffer_flush >= self.buffer_flush_interval:
                self._flush_buffer()
                self.last_buffer_flush = now
    
    def _flush_buffer(self):
        """Flush buffered events to callback."""
        if not self.event_buffer:
            return
            
        # Deduplicate similar events
        deduped_events = self._deduplicate_events(self.event_buffer)
        self.event_buffer.clear()
        
        # Send events to callback
        for event in deduped_events:
            try:
                self.callback(event)
            except Exception as e:
                print(f"Error in FS event callback: {e}")
    
    def _deduplicate_events(self, events: List[FSEvent]) -> List[FSEvent]:
        """Remove duplicate events from buffer."""
        # Group by path and event type
        event_groups: Dict[tuple, List[FSEvent]] = {}
        
        for event in events:
            key = (event.path, event.event_type)
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Keep only the most recent event for each path/type combination
        deduped = []
        for group in event_groups.values():
            # Sort by timestamp and take the latest
            latest_event = max(group, key=lambda e: e.timestamp)
            deduped.append(latest_event)
        
        return sorted(deduped, key=lambda e: e.timestamp)
    
    def start(self):
        """Start watching filesystem."""
        if self.running:
            return
            
        print(f"🔍 Starting FS watcher for {len(self.watch_paths)} paths")
        
        for path in self.watch_paths:
            if os.path.exists(path):
                print(f"   📁 Watching: {path}")
                self.observer.schedule(self.handler, path, recursive=self.recursive)
            else:
                print(f"   ⚠️ Path not found: {path}")
        
        self.observer.start()
        self.running = True
        
        # Start buffer flush thread
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()
    
    def _periodic_flush(self):
        """Periodically flush event buffer."""
        while self.running:
            time.sleep(self.buffer_flush_interval)
            with self.buffer_lock:
                if self.event_buffer:
                    self._flush_buffer()
                    self.last_buffer_flush = time.time()
    
    def stop(self):
        """Stop watching filesystem."""
        if not self.running:
            return
            
        print("🛑 Stopping FS watcher")
        self.running = False
        self.observer.stop()
        self.observer.join()
        
        # Flush any remaining events
        with self.buffer_lock:
            self._flush_buffer()
    
    def add_watch_path(self, path: str):
        """Add a new path to watch."""
        abs_path = os.path.abspath(path)
        if abs_path not in self.watch_paths:
            self.watch_paths.append(abs_path)
            if self.running and os.path.exists(abs_path):
                self.observer.schedule(self.handler, abs_path, recursive=self.recursive)
                print(f"📁 Added watch path: {abs_path}")
    
    def remove_watch_path(self, path: str):
        """Remove a path from watching."""
        abs_path = os.path.abspath(path)
        if abs_path in self.watch_paths:
            self.watch_paths.remove(abs_path)
            # Note: watchdog doesn't provide easy way to remove specific watches
            # Would need to restart observer for this
            print(f"🗑️ Removed watch path: {abs_path}")
    
    def get_watch_info(self) -> Dict[str, any]:
        """Get information about current watching state."""
        return {
            "running": self.running,
            "watch_paths": self.watch_paths,
            "ignore_patterns": self.ignore_patterns,
            "recursive": self.recursive,
            "buffer_size": len(self.event_buffer),
            "debounce_interval": self.debounce_interval,
            "buffer_flush_interval": self.buffer_flush_interval
        }


def create_project_fs_watcher(project_root: str, callback: Callable[[FSEvent], None]) -> FSWatcher:
    """Create an FSWatcher configured for a typical development project."""
    
    # Common development directories to watch
    watch_paths = [project_root]
    
    # Add subdirectories if they exist
    common_dirs = ['src', 'lib', 'app', 'components', 'pages', 'api', 'utils', 'config', 'tests', 'docs']
    for dirname in common_dirs:
        dirpath = os.path.join(project_root, dirname)
        if os.path.exists(dirpath):
            watch_paths.append(dirpath)
    
    # Project-specific ignore patterns
    project_ignore_patterns = FSWatcher.DEFAULT_IGNORE_PATTERNS + [
        # Language-specific
        '*.class', '*.jar', '*.war',  # Java
        '*.o', '*.so', '*.dll', '*.exe',  # C/C++
        '*.beam',  # Erlang
        # Framework-specific
        '.next/*', '.nuxt/*', '.svelte-kit/*',  # Frontend frameworks
        'venv/*', 'env/*', '.env/*',  # Python virtual environments
        'target/*',  # Rust, Scala
    ]
    
    return FSWatcher(
        watch_paths=watch_paths,
        callback=callback,
        ignore_patterns=project_ignore_patterns,
        recursive=True
    )
