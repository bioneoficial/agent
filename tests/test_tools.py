import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import _safe_run, read_file, write_file, append_file, generate_commit_message

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
            self.assertIn("dangerous", result.lower())
    
    @patch('subprocess.run')
    def test_rm_with_wildcard_runs_in_tempdir(self, mock_run):
        """Test that rm with wildcard runs in a temporary directory"""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        result = _safe_run(["rm", "-rf", "*.txt"])
        
        self.assertIn("WARNING", result)
        self.assertIn("temporary directory", result.lower())

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
    
    def test_write_and_read_file(self):
        """Test writing and reading a file"""
        test_content = "Hello, world!"
        test_file = "test.txt"
        
        write_file(f"{test_file}|{test_content}")
        content = read_file(test_file)
        
        self.assertEqual(content, test_content)
    
    def test_append_file(self):
        """Test appending to a file"""
        initial_content = "First line\n"
        append_content = "Second line"
        test_file = "append_test.txt"
        
        # Write initial content
        write_file(f"{test_file}|{initial_content}")
        
        # Append content
        append_file(f"{test_file}|{append_content}")
        
        # Read back
        content = read_file(test_file)
        self.assertEqual(content, initial_content + append_content)
        
    def test_append_to_nonexistent_file(self):
        """Test appending to a file that doesn't exist yet"""
        test_file = "new_file.txt"
        content = "New content"
        
        # Append to non-existent file should create it
        result = append_file(f"{test_file}|{content}")
        
        # Check result message
        self.assertIn("created", result.lower())
        
        # Verify content
        actual_content = read_file(test_file)
        self.assertEqual(actual_content, content)

class TestCommitMessageGenerator(unittest.TestCase):
    """Test the commit message generator"""
    
    def test_commit_message_from_diff_summary(self):
        """Test generating commit message from a diff summary"""
        diff_summary = """
 tools.py          | 120 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 agent_core.py     | 85  +++++++++++++++++++++++++++++++++++++++++
 requirements.txt  | 2   ++
        """
        
        message = generate_commit_message(diff_summary)
        
        # Message should contain the file names
        self.assertIn("tools.py", message)
        self.assertIn("agent_core.py", message)
        
        # Should identify as a feature (Python files)
        self.assertIn("feat", message)

if __name__ == "__main__":
    unittest.main() 