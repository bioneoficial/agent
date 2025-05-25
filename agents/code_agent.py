import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent

class CodeAgent(BaseAgent):
    """Specialized agent for code generation, file creation and editing"""
    
    def __init__(self):
        system_prompt = """You are an expert programmer specialized in code generation and file manipulation.

Your responsibilities:
1. Generate clean, functional code in any programming language
2. Create files with appropriate content based on descriptions
3. Edit existing files according to instructions
4. Support multiple programming languages with best practices

When generating code:
- Write clean, well-commented code
- Follow language-specific conventions
- Include necessary imports/dependencies
- Create complete, runnable code
- Add appropriate error handling

CRITICAL: Return ONLY the requested code/content. No explanations, no markdown fences, no thinking."""
        
        super().__init__("CodeAgent", system_prompt)
        
        # Language extension mapping
        self.lang_extensions = {
            'python': 'py', 'javascript': 'js', 'typescript': 'ts', 
            'java': 'java', 'c++': 'cpp', 'c#': 'cs', 'go': 'go',
            'rust': 'rs', 'ruby': 'rb', 'php': 'php', 'swift': 'swift',
            'kotlin': 'kt', 'scala': 'scala', 'r': 'r', 'matlab': 'm',
            'bash': 'sh', 'shell': 'sh', 'powershell': 'ps1'
        }
        
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the request"""
        request_lower = request.lower()
        code_keywords = [
            'criar arquivo', 'create file', 'crie arquivo', 'novo arquivo',
            'gerar código', 'generate code', 'escrever função', 'write function',
            'editar arquivo', 'edit file', 'modificar arquivo', 'modify file',
            'código', 'code', 'programa', 'script', 'função', 'function',
            'classe', 'class', 'método', 'method'
        ]
        
        # Check for file creation with extensions
        has_extension = bool(re.search(r'\.[a-zA-Z0-9]{1,4}\b', request))
        
        # Check for programming language mentions
        has_language = any(lang in request_lower for lang in self.lang_extensions.keys())
        
        return any(keyword in request_lower for keyword in code_keywords) or has_extension or has_language
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process code-related requests"""
        request_lower = request.lower()
        
        # Determine the type of operation
        if any(word in request_lower for word in ['criar', 'create', 'novo', 'new', 'gerar', 'generate']):
            return self._create_file(request)
        elif any(word in request_lower for word in ['editar', 'edit', 'modificar', 'modify', 'alterar', 'change']):
            return self._edit_file(request)
        elif any(word in request_lower for word in ['ler', 'read', 'mostrar', 'show', 'exibir', 'display']):
            return self._read_file(request)
        else:
            # Default to file creation if unclear
            return self._create_file(request)
    
    def _extract_filename(self, request: str) -> Optional[str]:
        """Extract filename from request"""
        # Try to find explicit filename with extension
        filename_match = re.search(r'(?:arquivo|file|chamado|named|nome)?\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,4})', request, re.IGNORECASE)
        if filename_match:
            return filename_match.group(1).strip()
        
        # Try to infer from language and generate filename
        for lang, ext in self.lang_extensions.items():
            if lang in request.lower():
                # Generate a descriptive filename
                timestamp = str(int(time.time()))[-6:]
                return f"generated_{timestamp}.{ext}"
        
        return None
    
    def _generate_code_content(self, request: str, filename: str) -> str:
        """Generate code content based on request"""
        # Extract language from filename
        ext = filename.split('.')[-1].lower()
        language = ext
        
        # Find language name for better prompting
        for lang, lang_ext in self.lang_extensions.items():
            if lang_ext == ext:
                language = lang
                break
        
        # Create specific prompt for code generation
        prompt = f"""Generate {language} code for the following request:

Request: {request}
Filename: {filename}

Requirements:
- Create complete, runnable code
- Include all necessary imports
- Add helpful comments
- Follow {language} best practices

Return ONLY the code content, nothing else:"""
        
        # Generate code
        code = self.invoke_llm(prompt, temperature=0.5)
        
        # Additional sanitization for code
        code = self.sanitize_llm_response(code)
        
        # Remove any remaining markdown fences
        code = re.sub(r'^```[a-zA-Z]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        
        return code
    
    def _create_file(self, request: str) -> Dict[str, Any]:
        """Create a new file with generated content"""
        # Extract filename
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Could not determine filename. Please specify a filename with extension (e.g., 'create file example.py')",
                "type": "file_creation"
            }
        
        # Generate content
        try:
            content = self._generate_code_content(request, filename)
            
            # Write file
            path = Path(filename)
            path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "output": f"Successfully created file '{filename}' with {len(content)} characters of generated code",
                "type": "file_creation",
                "filename": filename,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error creating file: {str(e)}",
                "type": "file_creation"
            }
    
    def _edit_file(self, request: str) -> Dict[str, Any]:
        """Edit an existing file"""
        # Extract filename
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Could not determine filename to edit",
                "type": "file_edit"
            }
        
        # Check if file exists
        path = Path(filename)
        if not path.exists():
            return {
                "success": False,
                "output": f"File '{filename}' does not exist",
                "type": "file_edit"
            }
        
        try:
            # Read current content
            current_content = path.read_text(encoding='utf-8')
            
            # Generate edit prompt
            prompt = f"""Edit the following code based on the request:

Current file content:
{current_content}

Edit request: {request}

Return the complete edited code, not just the changes:"""
            
            # Get edited content
            new_content = self.invoke_llm(prompt, temperature=0.3)
            new_content = self.sanitize_llm_response(new_content)
            
            # Remove markdown fences
            new_content = re.sub(r'^```[a-zA-Z]*\n', '', new_content)
            new_content = re.sub(r'\n```$', '', new_content)
            
            # Write updated content
            path.write_text(new_content, encoding='utf-8')
            
            return {
                "success": True,
                "output": f"Successfully edited file '{filename}'",
                "type": "file_edit",
                "filename": filename
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error editing file: {str(e)}",
                "type": "file_edit"
            }
    
    def _read_file(self, request: str) -> Dict[str, Any]:
        """Read file content"""
        filename = self._extract_filename(request)
        if not filename:
            return {
                "success": False,
                "output": "Could not determine filename to read",
                "type": "file_read"
            }
        
        path = Path(filename)
        if not path.exists():
            return {
                "success": False,
                "output": f"File '{filename}' does not exist",
                "type": "file_read"
            }
        
        try:
            content = path.read_text(encoding='utf-8')
            return {
                "success": True,
                "output": content,
                "type": "file_read",
                "filename": filename
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error reading file: {str(e)}",
                "type": "file_read"
            } 