"""
Trace storage utilities for saving and loading reasoning traces.
Manages the .orchestra/runs directory structure.
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from ..schemas.reasoning import ThoughtTrace, ReasoningConfig
import logging

logger = logging.getLogger(__name__)


class TraceStorage:
    """Manages storage and retrieval of reasoning traces."""
    
    def __init__(self, base_dir: str = ".orchestra"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore to avoid committing traces
        gitignore_path = self.base_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("# Orchestra reasoning traces\n")
                f.write("runs/\n")
                f.write("*.log\n")
                f.write("temp/\n")
    
    def create_run_directory(self, run_id: Optional[str] = None) -> str:
        """Create a new run directory and return the run ID."""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_id
    
    def save_trace(self, trace: ThoughtTrace, run_id: Optional[str] = None) -> str:
        """Save a reasoning trace to disk."""
        if run_id is None:
            run_id = self.create_run_directory()
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save main trace
        trace_path = run_dir / "trace.json"
        try:
            with open(trace_path, 'w', encoding='utf-8') as f:
                json.dump(trace.dict(), f, indent=2, default=str)
            
            # Save metadata
            metadata_path = run_dir / "metadata.json"
            metadata = {
                "run_id": run_id,
                "trace_id": trace.id,
                "goal": trace.goal,
                "created_at": trace.created_at.isoformat(),
                "reasoning_mode": trace.reasoning_mode,
                "step_count": len(trace.plan),
                "completion_rate": trace.get_completion_rate(),
                "overall_confidence": trace.overall_confidence
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved trace {trace.id} to run {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
            raise
    
    def load_trace(self, run_id: str) -> Optional[ThoughtTrace]:
        """Load a reasoning trace from disk."""
        trace_path = self.runs_dir / run_id / "trace.json"
        
        if not trace_path.exists():
            logger.warning(f"Trace file not found: {trace_path}")
            return None
        
        try:
            with open(trace_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ThoughtTrace(**data)
            
        except Exception as e:
            logger.error(f"Failed to load trace from {run_id}: {e}")
            return None
    
    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent runs with metadata."""
        runs = []
        
        try:
            for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
                if run_dir.is_dir() and len(runs) < limit:
                    metadata_path = run_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        runs.append(metadata)
                    else:
                        # Fallback for runs without metadata
                        runs.append({
                            "run_id": run_dir.name,
                            "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                            "goal": "Unknown",
                            "completion_rate": 0.0
                        })
        
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
        
        return runs
    
    def cleanup_old_runs(self, keep_days: int = 30):
        """Remove runs older than specified days."""
        cutoff = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        
        try:
            for run_dir in self.runs_dir.iterdir():
                if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(run_dir)
                    logger.info(f"Cleaned up old run: {run_dir.name}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
    
    def save_step_log(self, run_id: str, step_id: str, log_data: Dict[str, Any]):
        """Save detailed log for a specific step."""
        run_dir = self.runs_dir / run_id
        logs_dir = run_dir / "steps"
        logs_dir.mkdir(exist_ok=True)
        
        log_path = logs_dir / f"{step_id}.json"
        
        try:
            # Add timestamp
            log_data["timestamp"] = datetime.now().isoformat()
            log_data["step_id"] = step_id
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save step log: {e}")
    
    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific run."""
        trace = self.load_trace(run_id)
        if not trace:
            return None
        
        return {
            "run_id": run_id,
            "trace_id": trace.id,
            "goal": trace.goal,
            "created_at": trace.created_at.isoformat(),
            "reasoning_mode": trace.reasoning_mode,
            "total_steps": len(trace.plan),
            "completed_steps": len(trace.completed_steps),
            "failed_steps": len(trace.failed_steps),
            "completion_rate": f"{trace.get_completion_rate():.1%}",
            "overall_confidence": trace.overall_confidence,
            "active_risks": len(trace.get_active_risks()),
            "estimated_time": trace.estimated_total_time
        }
