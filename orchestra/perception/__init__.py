"""
Perception system for proactive monitoring and suggestions.
Includes filesystem watching, git monitoring, and suggestion generation.
"""

from .fs_watcher import FSWatcher
from .git_watcher import GitWatcher
from .perception_handler import PerceptionHandler

__all__ = ['FSWatcher', 'GitWatcher', 'PerceptionHandler']
