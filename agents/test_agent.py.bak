import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from agents.base_agent import BaseAgent

class TestAgent(BaseAgent):
    """Specialized agent for analyzing code and generating unit tests"""
    
    def __init__(self):
        system_prompt = """You are an expert in software testing and test-driven development.

Your responsibilities:
1. Analyze code files to understand functionality
2. Generate comprehensive unit tests with good coverage
3. Create tests for edge cases and error conditions
4. Use appropriate testing frameworks for each language

When generating tests:
- Write clear, descriptive test names
- Test both happy paths and edge cases
- Include setup and teardown when needed
- Mock external dependencies appropriately
- Aim for high code coverage

Testing frameworks by language:
- Python: pytest or unittest
- JavaScript/TypeScript: Jest or Mocha
- Java: JUnit
- C#: NUnit or xUnit
- Go: testing package
- Ruby: RSpec
- PHP: PHPUnit

CRITICAL: Return ONLY the test code. No explanations, no markdown fences, no thinking."""
        
        super().__init__("TestAgent", system_prompt)
        
        # Test framework mapping
        self.test_frameworks = {
            'py': 'pytest',
            'js': 'jest',
            'ts': 'jest',
            'java': 'junit',
            'cs': 'xunit',
            'go': 'testing',
            'rb': 'rspec',
            'php': 'phpunit',
            'cpp': 'gtest',
            'rs': 'cargo test'
        }
        
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the request"""
        request_lower = request.lower()
        test_keywords = [
            'test', 'teste', 'unit test', 'teste unitário',
            'testing', 'coverage', 'cobertura', 'test case',
            'caso de teste', 'testar', 'tests', 'testes'
        ]
        return any(keyword in request_lower for keyword in test_keywords)
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process test-related requests"""
        request_lower = request.lower()
        
        # Determine the type of test operation
        if any(word in request_lower for word in ['gerar', 'generate', 'criar', 'create', 'escrever', 'write']):
            return self._generate_tests(request)
        elif any(word in request_lower for word in ['analisar', 'analyze', 'análise', 'analysis']):
            return self._analyze_for_tests(request)
        else:
            # Default to generating tests
            return self._generate_tests(request)
    
    def _extract_files(self, request: str) -> List[str]:
        """Extract file names from request"""
        files = []
        
        # Look for explicit file names with extensions
        file_pattern = r'([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{1,4})'
        matches = re.findall(file_pattern, request)
        files.extend(matches)
        
        # If no explicit files, look for language indicators
        if not files:
            # Try to find all files of certain types in current directory
            for ext in ['py', 'js', 'ts', 'java', 'cs', 'go', 'rb', 'php']:
                if ext in request.lower() or self._get_language_name(ext).lower() in request.lower():
                    # Find files with this extension
                    for file in Path('.').glob(f'*.{ext}'):
                        files.append(str(file))
        
        return files
    
    def _get_language_name(self, extension: str) -> str:
        """Get language name from extension"""
        lang_map = {
            'py': 'python', 'js': 'javascript', 'ts': 'typescript',
            'java': 'java', 'cs': 'csharp', 'go': 'go',
            'rb': 'ruby', 'php': 'php', 'cpp': 'c++', 'rs': 'rust'
        }
        return lang_map.get(extension, extension)
    
    def _read_code_files(self, files: List[str]) -> Dict[str, str]:
        """Read content of code files"""
        contents = {}
        for file in files:
            path = Path(file)
            if path.exists() and path.is_file():
                try:
                    contents[file] = path.read_text(encoding='utf-8')
                except Exception as e:
                    contents[file] = f"Error reading file: {str(e)}"
        return contents
    
    def _generate_test_filename(self, source_file: str) -> str:
        """Generate appropriate test filename"""
        path = Path(source_file)
        name = path.stem
        ext = path.suffix
        
        # Common test file naming conventions
        if ext == '.py':
            return f"test_{name}.py"
        elif ext in ['.js', '.ts']:
            return f"{name}.test{ext}"
        elif ext == '.java':
            return f"{name}Test.java"
        elif ext == '.cs':
            return f"{name}Tests.cs"
        elif ext == '.go':
            return f"{name}_test.go"
        elif ext == '.rb':
            return f"{name}_spec.rb"
        elif ext == '.php':
            return f"{name}Test.php"
        else:
            return f"test_{name}{ext}"
    
    def _generate_tests(self, request: str) -> Dict[str, Any]:
        """Generate unit tests for specified files"""
        # Extract files to test
        files = self._extract_files(request)
        if not files:
            return {
                "success": False,
                "output": "No files specified. Please mention the file(s) to generate tests for.",
                "type": "test_generation"
            }
        
        # Read file contents
        file_contents = self._read_code_files(files)
        if not file_contents:
            return {
                "success": False,
                "output": f"Could not read any of the specified files: {', '.join(files)}",
                "type": "test_generation"
            }
        
        results = []
        
        for file, content in file_contents.items():
            if content.startswith("Error"):
                results.append(f"Skipped {file}: {content}")
                continue
            
            # Determine test framework
            ext = Path(file).suffix[1:]  # Remove the dot
            framework = self.test_frameworks.get(ext, 'generic')
            language = self._get_language_name(ext)
            
            # Generate test prompt
            prompt = f"""Generate comprehensive unit tests for the following {language} code:

File: {file}
Testing Framework: {framework}

Code to test:
{content}

Requirements:
- Test all public functions/methods
- Include edge cases and error conditions
- Use proper {framework} conventions
- Include necessary imports and setup
- Write descriptive test names
- Mock external dependencies if needed

Return ONLY the complete test code:"""
            
            # Generate tests
            test_code = self.invoke_llm(prompt, temperature=0.3)
            test_code = self.sanitize_llm_response(test_code)
            
            # Remove markdown fences
            test_code = re.sub(r'^```[a-zA-Z]*\n', '', test_code)
            test_code = re.sub(r'\n```$', '', test_code)
            
            # Generate test filename
            test_filename = self._generate_test_filename(file)
            
            # Write test file
            try:
                test_path = Path(test_filename)
                test_path.write_text(test_code, encoding='utf-8')
                results.append(f"✓ Generated tests for {file} -> {test_filename}")
            except Exception as e:
                results.append(f"✗ Failed to write tests for {file}: {str(e)}")
        
        return {
            "success": len([r for r in results if r.startswith('✓')]) > 0,
            "output": "\n".join(results),
            "type": "test_generation",
            "files_tested": len([r for r in results if r.startswith('✓')])
        }
    
    def _analyze_for_tests(self, request: str) -> Dict[str, Any]:
        """Analyze code files to suggest what tests are needed"""
        files = self._extract_files(request)
        if not files:
            return {
                "success": False,
                "output": "No files specified for analysis.",
                "type": "test_analysis"
            }
        
        file_contents = self._read_code_files(files)
        if not file_contents:
            return {
                "success": False,
                "output": f"Could not read any of the specified files: {', '.join(files)}",
                "type": "test_analysis"
            }
        
        analysis_prompt = f"""Analyze the following code files and suggest what unit tests should be created:

Files to analyze:
"""
        for file, content in file_contents.items():
            if not content.startswith("Error"):
                analysis_prompt += f"\n\nFile: {file}\n{content[:1000]}...\n"
        
        analysis_prompt += """
Provide a structured analysis of:
1. What functions/methods need testing
2. Important edge cases to cover
3. Potential error conditions
4. Recommended test structure

Be specific and actionable:"""
        
        analysis = self.invoke_llm(analysis_prompt, temperature=0.3)
        
        return {
            "success": True,
            "output": analysis,
            "type": "test_analysis",
            "files_analyzed": len([c for c in file_contents.values() if not c.startswith("Error")])
        } 