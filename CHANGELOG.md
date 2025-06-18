# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Consolidated FileAgent and TestAgent into a single CodeAgent for better maintainability
- Added comprehensive integration tests for end-to-end workflows
- Enhanced documentation for all key methods in CodeAgent
- Added CHANGELOG.md to track project changes

### Changed
- Updated Orchestrator to use the consolidated CodeAgent
- Improved error handling and user feedback
- Updated README.md to reflect the new architecture

### Removed
- Removed deprecated FileAgent and TestAgent files
- Removed legacy test files (test_exemplo.py, test_tools.py) that were no longer needed

## [0.1.0] - 2025-06-18

### Added
- Initial release of Git Terminal Assistant
- Basic agent architecture with GitAgent, ChatAgent, and CodeAgent
- Configuration system for per-agent model selection
- Terminal command routing and execution
