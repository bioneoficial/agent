"""
CLI notifier for user interaction with perception suggestions.
Handles displaying suggestions and capturing user responses.
"""

import os
import sys
import threading
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
from queue import Queue
import json

from .perception_handler import Suggestion


class CLINotifier:
    """CLI notifier for perception suggestions with user interaction."""
    
    def __init__(self, 
                 acceptance_callback: Optional[Callable[[str, str], None]] = None,
                 rejection_callback: Optional[Callable[[str, str], None]] = None):
        """
        Initialize CLI notifier.
        
        Args:
            acceptance_callback: Called when user accepts a suggestion (suggestion_id, action)
            rejection_callback: Called when user rejects a suggestion (suggestion_id, reason)
        """
        self.acceptance_callback = acceptance_callback
        self.rejection_callback = rejection_callback
        
        # Suggestion queue and management
        self.suggestion_queue = Queue()
        self.active_suggestions = {}  # id -> suggestion
        self.suggestion_history = []  # List of processed suggestions
        
        # Display settings
        self.silent_mode = os.getenv('GTA_PERCEPTION_SILENT', '0') == '1'
        self.auto_dismiss_minutes = int(os.getenv('GTA_PERCEPTION_AUTO_DISMISS', '30'))
        self.max_concurrent = int(os.getenv('GTA_PERCEPTION_MAX_CONCURRENT', '3'))
        
        # Threading
        self.running = False
        self.notifier_thread = None
        self.cleanup_thread = None
        
        # User interaction
        self.pending_input = False
        self.input_lock = threading.Lock()
    
    def start(self):
        """Start the CLI notifier system."""
        if self.running:
            return
        
        if self.silent_mode:
            print("🔇 Perception notifications in silent mode")
            return
        
        print("📢 Starting perception CLI notifier")
        self.running = True
        
        # Start notification thread
        self.notifier_thread = threading.Thread(target=self._notification_loop, daemon=True)
        self.notifier_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def stop(self):
        """Stop the CLI notifier system."""
        if not self.running:
            return
        
        print("🛑 Stopping perception CLI notifier")
        self.running = False
        
        if self.notifier_thread:
            self.notifier_thread.join(timeout=2)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2)
    
    def add_suggestion(self, suggestion: Suggestion):
        """Add a new suggestion to the notification queue."""
        if self.silent_mode:
            return
        
        # Check if we're already at max concurrent suggestions
        if len(self.active_suggestions) >= self.max_concurrent:
            print(f"⚠️ Suggestion queue full, dropping: {suggestion.title}")
            return
        
        self.suggestion_queue.put(suggestion)
    
    def _notification_loop(self):
        """Main notification processing loop."""
        while self.running:
            try:
                # Get next suggestion (with timeout to allow loop to exit)
                try:
                    suggestion = self.suggestion_queue.get(timeout=1)
                except:
                    continue
                
                if not self.running:
                    break
                
                # Display the suggestion
                self._display_suggestion(suggestion)
                self.active_suggestions[suggestion.id] = suggestion
                
            except Exception as e:
                print(f"Error in notification loop: {e}")
    
    def _cleanup_loop(self):
        """Cleanup expired suggestions."""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                
                if not self.running:
                    break
                
                now = datetime.now()
                expired_suggestions = []
                
                for suggestion_id, suggestion in self.active_suggestions.items():
                    # Check if suggestion has expired
                    if suggestion.expires_at and now > suggestion.expires_at:
                        expired_suggestions.append(suggestion_id)
                    # Check auto-dismiss timeout
                    elif (now - suggestion.created_at).total_seconds() > self.auto_dismiss_minutes * 60:
                        expired_suggestions.append(suggestion_id)
                
                # Remove expired suggestions
                for suggestion_id in expired_suggestions:
                    suggestion = self.active_suggestions.pop(suggestion_id, None)
                    if suggestion:
                        self.suggestion_history.append({
                            'suggestion': suggestion,
                            'status': 'expired',
                            'timestamp': datetime.now()
                        })
                        print(f"⏰ Suggestion expired: {suggestion.title}")
                
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
    
    def _display_suggestion(self, suggestion: Suggestion):
        """Display a suggestion to the user."""
        with self.input_lock:
            print("\n" + "="*80)
            print(f"🧠 PERCEPTION SUGGESTION [{suggestion.priority.upper()}]")
            print("="*80)
            print(f"📝 {suggestion.title}")
            print(f"💡 {suggestion.description}")
            print(f"🤔 Reasoning: {suggestion.reasoning}")
            
            if suggestion.suggested_actions:
                print(f"🎯 Suggested Actions:")
                for i, action in enumerate(suggestion.suggested_actions, 1):
                    print(f"   {i}. {action}")
            
            if suggestion.tags:
                print(f"🏷️  Tags: {', '.join(suggestion.tags)}")
            
            print(f"🆔 ID: {suggestion.id}")
            print("="*80)
            print("Actions:")
            print("  [a] Accept and execute suggestions")
            print("  [d] Dismiss this suggestion")
            print("  [s] Show more details")
            print("  [l] List all active suggestions")
            print("  [h] Help with commands")
            print("  [Enter] Continue without action")
            print("="*80)
            
            # Note: We don't block for input here to avoid blocking the main thread
            # Instead, we provide interactive commands that users can run separately
    
    def handle_user_input(self, user_input: str) -> Dict[str, any]:
        """
        Handle user input for suggestion management.
        This can be called from the main GTA command interface.
        """
        if not user_input or not user_input.strip():
            return {"status": "no_action"}
        
        command = user_input.strip().lower()
        
        if command == 'l' or command == 'list':
            return self._list_active_suggestions()
        
        elif command == 'h' or command == 'help':
            return self._show_help()
        
        elif command == 'a' or command == 'accept':
            # Accept most recent suggestion: "a" or "accept"
            return self._accept_latest_suggestion()
        
        elif command.startswith('a ') or command.startswith('accept '):
            # Accept suggestion: "accept suggestion_id" or "a suggestion_id"
            parts = command.split(' ', 1)
            if len(parts) > 1:
                return self._accept_suggestion(parts[1])
            else:
                return {"status": "error", "message": "Please specify suggestion ID to accept"}
        
        elif command == 'd' or command == 'dismiss':
            # Dismiss most recent suggestion: "d" or "dismiss"
            return self._dismiss_latest_suggestion()
        
        elif command.startswith('d ') or command.startswith('dismiss '):
            # Dismiss specific suggestion: "dismiss suggestion_id" or "d suggestion_id"
            parts = command.split(' ', 1)
            if len(parts) > 1:
                return self._dismiss_suggestion(parts[1])
            else:
                return {"status": "error", "message": "Please specify suggestion ID to dismiss"}
        
        elif command.startswith('s ') or command.startswith('show '):
            # Show details: "show suggestion_id" or "s suggestion_id"
            parts = command.split(' ', 1)
            if len(parts) > 1:
                return self._show_suggestion_details(parts[1])
            else:
                return {"status": "error", "message": "Please specify suggestion ID to show"}
        
        else:
            return {"status": "unknown_command", "message": f"Unknown command: {command}"}
    
    def _list_active_suggestions(self) -> Dict[str, any]:
        """List all active suggestions."""
        if not self.active_suggestions:
            return {
                "status": "info",
                "message": "No active suggestions",
                "suggestions": []
            }
        
        suggestions = []
        for suggestion in self.active_suggestions.values():
            suggestions.append({
                "id": suggestion.id,
                "title": suggestion.title,
                "priority": suggestion.priority,
                "type": suggestion.type,
                "created_at": suggestion.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "tags": suggestion.tags
            })
        
        return {
            "status": "success",
            "message": f"Found {len(suggestions)} active suggestions",
            "suggestions": suggestions
        }
    
    def _show_help(self) -> Dict[str, any]:
        """Show help for suggestion commands."""
        help_text = """
Perception Suggestion Commands:
  list, l                    - List all active suggestions
  accept <id>, a <id>        - Accept and execute suggestion
  dismiss <id>, d <id>       - Dismiss suggestion
  show <id>, s <id>          - Show detailed suggestion info
  help, h                    - Show this help
  
Examples:
  list
  accept perception_1_1234567890
  dismiss perception_2_1234567891
  show perception_1_1234567890
"""
        return {
            "status": "info",
            "message": help_text.strip(),
            "suggestions": []
        }
    
    def _accept_suggestion(self, suggestion_id: str) -> Dict[str, any]:
        """Accept and execute a suggestion."""
        suggestion_id = suggestion_id.strip()
        suggestion = self.active_suggestions.get(suggestion_id)
        
        if not suggestion:
            return {
                "status": "error",
                "message": f"Suggestion {suggestion_id} not found"
            }
        
        # Remove from active suggestions
        self.active_suggestions.pop(suggestion_id)
        
        # Add to history
        self.suggestion_history.append({
            'suggestion': suggestion,
            'status': 'accepted',
            'timestamp': datetime.now()
        })
        
        # Call acceptance callback if provided
        if self.acceptance_callback:
            try:
                # Pass the first suggested action as the default action
                default_action = suggestion.suggested_actions[0] if suggestion.suggested_actions else ""
                self.acceptance_callback(suggestion_id, default_action)
            except Exception as e:
                print(f"Error in acceptance callback: {e}")
        
        return {
            "status": "success",
            "message": f"Accepted suggestion: {suggestion.title}",
            "suggestion": {
                "id": suggestion.id,
                "title": suggestion.title,
                "actions": suggestion.suggested_actions
            }
        }
    
    def _accept_latest_suggestion(self) -> Dict[str, any]:
        """Accept the most recent active suggestion."""
        if not self.active_suggestions:
            return {
                "status": "error", 
                "message": "No active suggestions to accept"
            }
        
        # Get the most recent suggestion (by creation time)
        latest_suggestion = max(self.active_suggestions.values(), 
                              key=lambda s: s.created_at)
        return self._accept_suggestion(latest_suggestion.id)
    
    def _dismiss_latest_suggestion(self) -> Dict[str, any]:
        """Dismiss the most recent active suggestion."""
        if not self.active_suggestions:
            return {
                "status": "error",
                "message": "No active suggestions to dismiss" 
            }
        
        # Get the most recent suggestion (by creation time)
        latest_suggestion = max(self.active_suggestions.values(),
                              key=lambda s: s.created_at)
        return self._dismiss_suggestion(latest_suggestion.id)

    def _dismiss_suggestion(self, suggestion_id: str) -> Dict[str, any]:
        """Dismiss a suggestion."""
        suggestion_id = suggestion_id.strip()
        suggestion = self.active_suggestions.get(suggestion_id)
        
        if not suggestion:
            return {
                "status": "error",
                "message": f"Suggestion {suggestion_id} not found"
            }
        
        # Remove from active suggestions
        self.active_suggestions.pop(suggestion_id)
        
        # Add to history
        self.suggestion_history.append({
            'suggestion': suggestion,
            'status': 'dismissed',
            'timestamp': datetime.now()
        })
        
        # Call rejection callback if provided
        if self.rejection_callback:
            try:
                self.rejection_callback(suggestion_id, "user_dismissed")
            except Exception as e:
                print(f"Error in rejection callback: {e}")
        
        return {
            "status": "success",
            "message": f"Dismissed suggestion: {suggestion.title}"
        }
    
    def _show_suggestion_details(self, suggestion_id: str) -> Dict[str, any]:
        """Show detailed information about a suggestion."""
        suggestion_id = suggestion_id.strip()
        suggestion = self.active_suggestions.get(suggestion_id)
        
        if not suggestion:
            return {
                "status": "error",
                "message": f"Suggestion {suggestion_id} not found"
            }
        
        details = {
            "id": suggestion.id,
            "type": suggestion.type,
            "priority": suggestion.priority,
            "title": suggestion.title,
            "description": suggestion.description,
            "reasoning": suggestion.reasoning,
            "suggested_actions": suggestion.suggested_actions,
            "context": suggestion.context,
            "created_at": suggestion.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "expires_at": suggestion.expires_at.strftime("%Y-%m-%d %H:%M:%S") if suggestion.expires_at else None,
            "tags": suggestion.tags
        }
        
        return {
            "status": "success",
            "message": f"Suggestion details for {suggestion_id}",
            "suggestion": details
        }
    
    def get_status(self) -> Dict[str, any]:
        """Get notifier status."""
        return {
            "running": self.running,
            "silent_mode": self.silent_mode,
            "active_suggestions": len(self.active_suggestions),
            "suggestion_history": len(self.suggestion_history),
            "max_concurrent": self.max_concurrent,
            "auto_dismiss_minutes": self.auto_dismiss_minutes
        }
    
    def clear_all_suggestions(self) -> Dict[str, any]:
        """Clear all active suggestions."""
        count = len(self.active_suggestions)
        
        # Move all to history as dismissed
        for suggestion in self.active_suggestions.values():
            self.suggestion_history.append({
                'suggestion': suggestion,
                'status': 'cleared',
                'timestamp': datetime.now()
            })
        
        self.active_suggestions.clear()
        
        return {
            "status": "success",
            "message": f"Cleared {count} suggestions"
        }
    
    def get_suggestion_summary(self) -> Dict[str, any]:
        """Get a summary of suggestion activity."""
        # Count by status
        status_counts = {}
        for entry in self.suggestion_history:
            status = entry['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by type
        type_counts = {}
        for suggestion in self.active_suggestions.values():
            suggestion_type = suggestion.type
            type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
        
        for entry in self.suggestion_history:
            suggestion_type = entry['suggestion'].type
            if suggestion_type not in type_counts:
                type_counts[suggestion_type] = 0
        
        return {
            "active_count": len(self.active_suggestions),
            "history_count": len(self.suggestion_history),
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "running": self.running,
            "silent_mode": self.silent_mode
        }


def create_perception_cli_integration():
    """Create CLI notifier with integration hooks for GTA."""
    
    def on_suggestion_accepted(suggestion_id: str, action: str):
        """Handle suggestion acceptance."""
        print(f"✅ Executing suggestion: {action}")
        # This could trigger the planner/executor to create a plan for the action
        # For now, just log it
    
    def on_suggestion_rejected(suggestion_id: str, reason: str):
        """Handle suggestion rejection."""
        print(f"❌ Suggestion dismissed: {reason}")
        # Could be used for learning user preferences
    
    return CLINotifier(
        acceptance_callback=on_suggestion_accepted,
        rejection_callback=on_suggestion_rejected
    )
