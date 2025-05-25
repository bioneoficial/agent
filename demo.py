#!/usr/bin/env python3
"""
Demo script to showcase Git Terminal Assistant capabilities
"""

import subprocess
import time

def run_command(cmd):
    """Run a GTA command and show the result"""
    print(f"\n{'='*60}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(
        f"python main.py {cmd}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    time.sleep(1)  # Pause for readability
    return result.returncode == 0

def main():
    print("Git Terminal Assistant - Demo")
    print("="*60)
    
    # Show available agents
    print("\n1. Showing available agents and their capabilities:")
    run_command("agents")
    
    # Test 1: Create a Python file with code
    print("\n2. Testing Code Agent - Creating a Python file:")
    run_command("criar arquivo calculator.py com funções de soma, subtração, multiplicação e divisão")
    
    # Test 2: Generate tests for the file
    print("\n3. Testing Test Agent - Generating unit tests:")
    run_command("gerar testes para calculator.py")
    
    # Test 3: Show help
    print("\n4. Showing help system:")
    run_command("help")
    
    print("\n" + "="*60)
    print("Demo completed! Key features demonstrated:")
    print("✓ Clean code generation without artifacts")
    print("✓ Specialized test generation with proper framework")
    print("✓ Intelligent commit messages")
    print("✓ Clear help and agent information")
    print("="*60)

if __name__ == "__main__":
    main() 