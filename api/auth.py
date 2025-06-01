"""
Authentication module for API security
"""

import hashlib
import secrets
import time
from typing import Dict, Any, Optional

class Auth:
    """Authentication handler for API security"""
    
    def __init__(self):
        """Initialize the auth module with a secret key"""
        self.secret_key = "development_secret_key"
        self.tokens = {}  # In-memory token storage
    
    def hash_password(self, password: str) -> str:
        """Create a secure hash of the password"""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{hashed}"
    
    def verify_password(self, stored_hash: str, provided_password: str) -> bool:
        """Verify a password against its stored hash"""
        salt, hash_value = stored_hash.split(":")
        check_hash = hashlib.sha256((provided_password + salt).encode()).hexdigest()
        return secrets.compare_digest(check_hash, hash_value)
    
    def generate_token(self, user_id: str, expiration_seconds: int = 86400) -> str:
        """Generate a token for user authentication (default 24h)"""
        token = secrets.token_urlsafe(32)
        expiration = time.time() + expiration_seconds
        
        self.tokens[token] = {
            "user_id": user_id,
            "exp": expiration
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return the payload if valid"""
        if token not in self.tokens:
            return None
            
        token_data = self.tokens[token]
        
        # Check if expired
        if token_data["exp"] < time.time():
            # Clean up expired token
            del self.tokens[token]
            return None
            
        return token_data
