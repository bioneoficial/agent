import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import _safe_run, create_file, edit_file, remove_file, commit_auto, commit_staged, git_status

class TestSafetyFeatures(unittest.TestCase):
    """Test the safety features of tools.py"""
    
    def test_dangerous_command_blocked(self):
        """Test that dangerous commands are blocked"""
        dangerous_commands = [
            ["rm", "-rf", "/"],
            ["dd", "if=/dev/random", "of=/dev/sda"],
            ["mkfs", "/dev/sda1"],
            [":(){:|:&};:", ":"],
        ]
        
        for cmd in dangerous_commands:
            result = _safe_run(cmd)
            self.assertIn("BLOCKED", result)
            self.assertIn("perigoso", result.lower())
    
    @patch('subprocess.run')
    def test_safe_command_allowed(self, mock_run):
        """Test that safe commands are allowed"""
        mock_run.return_value = MagicMock(returncode=0, stdout="test output", stderr="")
        
        result = _safe_run(["ls", "-la"])
        self.assertEqual(result, "test output")

class TestFileOperations(unittest.TestCase):
    """Test file operations"""
    
    def setUp(self):
        """Set up temporary directory for file tests"""
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tempdir.name)
    
    def tearDown(self):
        """Clean up after tests"""
        os.chdir(self.old_cwd)
        self.tempdir.cleanup()
    
    def test_create_file(self):
        """Test creating a file with content"""
        test_content = "Hello, world!"
        test_file = "test.txt"
        
        result = create_file(f"{test_file}|{test_content}")
        self.assertIn("sucesso", result.lower())
        
        with open(test_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, test_content)
    
    def test_edit_file(self):
        """Test editing an existing file"""
        initial_content = "Initial content"
        new_content = "New content"
        test_file = "edit_test.txt"
        
        # Create the file first
        create_file(f"{test_file}|{initial_content}")
        
        # Edit the file
        result = edit_file(f"{test_file}|{new_content}")
        self.assertIn("sucesso", result.lower())
        
        # Verify content was updated
        with open(test_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, new_content)
    
    def test_remove_file(self):
        """Test removing a file"""
        test_file = "remove_test.txt"
        
        # Create a file to remove
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Remove the file
        result = remove_file(test_file)
        self.assertIn("sucesso", result.lower())
        
        # Verify file was removed
        self.assertFalse(os.path.exists(test_file))
    
    def test_remove_nonexistent_file(self):
        """Test attempting to remove a file that doesn't exist"""
        nonexistent_file = "nonexistent.txt"
        
        result = remove_file(nonexistent_file)
        self.assertIn("não encontrado", result.lower())
    
    def test_markdown_content_sanitization(self):
        """Test that markdown fences are properly removed from content"""
        markdown_content = "```python\nprint('Hello')\n```"
        expected_content = "print('Hello')"
        test_file = "markdown_test.txt"
        
        create_file(f"{test_file}|{markdown_content}")
        
        with open(test_file, 'r') as f:
            content = f.read().strip()
        self.assertEqual(content, expected_content)

@patch('tools.git_status')
@patch('tools.generate_commit_message')
class TestGitOperations(unittest.TestCase):
    """Test Git operations"""
    
    def test_commit_staged_with_files(self, mock_generate_message, mock_git_status):
        """Test committing staged files"""
        # Mock git status to return staged files
        mock_git_status.side_effect = lambda cmd: "file1.txt\nfile2.txt" if "diff --cached" in cmd else "commit message"
        
        result = commit_staged()
        
        # Verify git commit was called
        mock_git_status.assert_any_call('commit -m "chore: commit staged changes"')
        self.assertEqual(result, "commit message")
    
    def test_commit_staged_no_files(self, mock_generate_message, mock_git_status):
        """Test attempting to commit with no staged files"""
        # Mock git status to return no staged files
        mock_git_status.side_effect = lambda cmd: "" if "diff --cached" in cmd else None
        
        result = commit_staged()
        
        # Verify no commit was attempted
        self.assertIn("Nenhum arquivo", result)
    
    def test_commit_auto_with_message_generation(self, mock_generate_message, mock_git_status):
        """Test auto-committing with generated message"""
        # Set up mocks
        mock_git_status.side_effect = lambda cmd: "file1.txt" if "diff --cached" in cmd else "diff stats"
        mock_generate_message.return_value = "feat: auto-generated commit message"
        
        # Patch _safe_run to avoid actual git commands
        with patch('tools._safe_run') as mock_safe_run:
            mock_safe_run.return_value = "Committed"
            
            # Test with stage_all=True
            result = commit_auto(stage_all=True)
            
            # Verify add and commit were called
            mock_safe_run.assert_any_call(["git", "add", "-A"])
            mock_safe_run.assert_any_call(["git", "commit", "-m", "feat: auto-generated commit message"])
            self.assertEqual(result, "Committed")

if __name__ == "__main__":
    unittest.main() 