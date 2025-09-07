#!/usr/bin/env python3
"""
Simple test script to verify composite request detection works properly
with both Portuguese and English patterns.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import just the detection logic without the full agent system
def create_test_planner():
    """Create a minimal planner with just the detection patterns."""
    class TestPlanner:
        def __init__(self):
            # Composite request indicators (Portuguese and English)
            self.composite_indicators = [
                # Portuguese patterns
                r'criar.*(?:com|e|mais).*(?:test|doc|estrutura)',
                r'fazer.*(?:projeto|módulo|sistema).*(?:com|e|incluir)',
                r'desenvolver.*(?:completo|com|incluindo)',
                r'implementar.*(?:com|incluindo|e).*(?:test|doc)',
                r'configurar.*(?:projeto|sistema).*(?:com|incluindo)',
                r'preparar.*(?:ambiente|projeto).*(?:com|para)',
                # English patterns
                r'create.*(?:with|and|plus).*(?:test|doc|structure)',
                r'make.*(?:project|module|system).*(?:with|and|include)',
                r'develop.*(?:complete|with|including)',
                r'implement.*(?:with|including|and).*(?:test|doc)',
                r'configure.*(?:project|system).*(?:with|including)',
                r'prepare.*(?:environment|project).*(?:with|for)',
                r'build.*(?:with|and|including).*(?:test|doc)',
                r'setup.*(?:with|and|including)',
                r'generate.*(?:with|and|including)'
            ]
            
            # Task detection patterns
            self.task_patterns = {
                'file_create': [
                    r'(?:criar|gerar|fazer).*(?:arquivo|file)',
                    r'novo\s+arquivo',
                    r'escrever.*(?:em|no)\s+arquivo'
                ],
                'file_edit': [
                    r'(?:editar|modificar|alterar|mudar).*arquivo',
                    r'adicionar.*(?:em|no|ao)\s+arquivo',
                    r'atualizar.*arquivo'
                ],
                'test_run': [
                    r'(?:executar|rodar|fazer).*test',
                    r'testar.*(?:código|função|projeto)',
                    r'verificar.*test'
                ],
                'test_generate': [
                    r'(?:criar|gerar|escrever).*test',
                    r'test.*para.*(?:função|código)',
                    r'adicionar.*test'
                ],
                'git_commit': [
                    r'(?:fazer|gerar|criar).*commit',
                    r'commitar.*(?:mudanças|código|arquivo)',
                    r'salvar.*(?:no\s+)?git'
                ]
            }
        
        def is_composite_request(self, request: str) -> bool:
            """Determine if a request contains multiple tasks."""
            request_lower = request.lower()
            
            # Check for composite indicators
            for pattern in self.composite_indicators:
                if re.search(pattern, request_lower):
                    return True
            
            # Check for multiple task types
            detected_tasks = []
            for task_type, patterns in self.task_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, request_lower):
                        detected_tasks.append(task_type)
                        break
            
            return len(detected_tasks) > 1
    
    return TestPlanner()

def test_composite_detection():
    """Test composite request detection with various patterns."""
    planner = create_test_planner()
    
    # Test cases: (request, expected_result, description)
    test_cases = [
        # Portuguese patterns
        ("criar um projeto com testes e documentação", True, "Portuguese: create project with tests and docs"),
        ("fazer um módulo e incluir testes", True, "Portuguese: make module and include tests"),
        ("implementar função com testes unitários", True, "Portuguese: implement function with unit tests"),
        ("configurar projeto com estrutura completa", True, "Portuguese: configure project with complete structure"),
        
        # English patterns  
        ("create a project with tests and documentation", True, "English: create project with tests and docs"),
        ("build a module and include unit tests", True, "English: build module and include tests"),
        ("implement a function with tests and docs", True, "English: implement function with tests and docs"),
        ("setup a complete project structure", True, "English: setup complete project structure"),
        ("develop a system including tests", True, "English: develop system including tests"),
        ("generate code with documentation", True, "English: generate code with documentation"),
        
        # Single task requests (should be False)
        ("criar um arquivo", False, "Portuguese: create single file"),
        ("create a file", False, "English: create single file"),
        ("executar testes", False, "Portuguese: run tests only"),
        ("run tests", False, "English: run tests only"),
        ("fazer commit", False, "Portuguese: make commit only"),
        ("git commit", False, "English: git commit only"),
        
        # Edge cases
        ("", False, "Empty string"),
        ("hello world", False, "Simple greeting"),
    ]
    
    print("🧪 Testing Composite Request Detection")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for request, expected, description in test_cases:
        try:
            result = planner.is_composite_request(request)
            status = "✅ PASS" if result == expected else "❌ FAIL"
            
            if result == expected:
                passed += 1
            else:
                failed += 1
                
            print(f"{status} {description}")
            print(f"   Request: '{request}' → Detected: {result} (Expected: {expected})")
            print()
            
        except Exception as e:
            failed += 1
            print(f"❌ ERROR {description}")
            print(f"   Request: '{request}' → Error: {str(e)}")
            print()
    
    print("=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = test_composite_detection()
    sys.exit(0 if success else 1)
