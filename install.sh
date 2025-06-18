#!/bin/bash

# Git Terminal Assistant - Installation Script
# This script sets up the GTA command globally

set -e

echo "🚀 Installing Git Terminal Assistant (GTA)..."

# Get current directory (project root)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GTA_SCRIPT="$PROJECT_DIR/gta"

# Make the gta script executable
chmod +x "$GTA_SCRIPT"

echo "✅ Made gta script executable"

# Determine the shell and profile file
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    "bash")
        PROFILE_FILE="$HOME/.bashrc"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            PROFILE_FILE="$HOME/.bash_profile"
        fi
        ;;
    "zsh")
        PROFILE_FILE="$HOME/.zshrc"
        ;;
    *)
        PROFILE_FILE="$HOME/.profile"
        ;;
esac

echo "📝 Detected shell: $SHELL_NAME"
echo "📝 Using profile file: $PROFILE_FILE"

# Create alias in shell profile
ALIAS_LINE="alias gta='$GTA_SCRIPT'"

# Check if alias already exists
if grep -q "alias gta=" "$PROFILE_FILE" 2>/dev/null; then
    echo "⚠️  GTA alias already exists in $PROFILE_FILE"
    echo "🔄 Updating existing alias..."
    # Remove old alias and add new one
    grep -v "alias gta=" "$PROFILE_FILE" > "$PROFILE_FILE.tmp" && mv "$PROFILE_FILE.tmp" "$PROFILE_FILE"
fi

# Add new alias
echo "" >> "$PROFILE_FILE"
echo "# Git Terminal Assistant" >> "$PROFILE_FILE"
echo "$ALIAS_LINE" >> "$PROFILE_FILE"

echo "✅ Added GTA alias to $PROFILE_FILE"

# Optional: Add to /usr/local/bin for system-wide access
read -p "🤔 Install system-wide? (requires sudo) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo ln -sf "$GTA_SCRIPT" /usr/local/bin/gta
    echo "✅ Created system-wide link at /usr/local/bin/gta"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Usage:"
echo "   gta -i                    # Interactive mode"
echo "   gta -i -m ask            # Ask mode"
echo "   gta -i -m free           # Free mode"
echo "   gta 'commit my changes'   # Single command"
echo "   gta 'git status'         # Git command"
echo ""
echo "🔄 Reload your shell or run: source $PROFILE_FILE"
echo "   Then you can use 'gta' from anywhere!" 