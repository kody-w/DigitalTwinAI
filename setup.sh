#!/bin/bash

# C365Tester - Installation and Setup Script
# This script sets up the development environment for the Azure Function

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   C365Tester - Environment Setup Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Step 1: Install Python 3.11
echo -e "${BLUE}Step 1: Installing Python 3.11...${NC}"
if command_exists python3.11; then
    print_status "Python 3.11 already installed ($(python3.11 --version))"
else
    print_info "Installing Python 3.11..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y >/dev/null 2>&1
    sudo apt-get update >/dev/null 2>&1
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev >/dev/null 2>&1
    
    if command_exists python3.11; then
        print_status "Python 3.11 installed successfully ($(python3.11 --version))"
    else
        print_error "Failed to install Python 3.11"
        exit 1
    fi
fi
echo

# Step 2: Install Node.js (if not present)
echo -e "${BLUE}Step 2: Checking Node.js...${NC}"
if command_exists node; then
    print_status "Node.js already installed ($(node --version))"
else
    print_error "Node.js not found. Please install Node.js first."
    exit 1
fi
echo

# Step 3: Install Azure Functions Core Tools
echo -e "${BLUE}Step 3: Installing Azure Functions Core Tools...${NC}"
if command_exists func; then
    print_status "Azure Functions Core Tools already installed (version $(func --version))"
else
    print_info "Installing Azure Functions Core Tools..."
    npm install -g azure-functions-core-tools@4 --unsafe-perm true >/dev/null 2>&1
    
    if command_exists func; then
        print_status "Azure Functions Core Tools installed successfully (version $(func --version))"
    else
        print_error "Failed to install Azure Functions Core Tools"
        exit 1
    fi
fi
echo

# Step 4: Create Python virtual environment
echo -e "${BLUE}Step 4: Setting up Python virtual environment...${NC}"
if [ -d ".venv" ]; then
    print_info "Virtual environment already exists, recreating..."
    rm -rf .venv
fi

print_info "Creating virtual environment..."
python3.11 -m venv .venv

if [ -d ".venv" ]; then
    print_status "Virtual environment created successfully"
else
    print_error "Failed to create virtual environment"
    exit 1
fi
echo

# Step 5: Activate virtual environment and install dependencies
echo -e "${BLUE}Step 5: Installing Python dependencies...${NC}"
source .venv/bin/activate

print_info "Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1

print_info "Installing required packages..."
pip install azure-functions==1.18.0 >/dev/null 2>&1
pip install openai==1.55.3 >/dev/null 2>&1
pip install azure-storage-file==2.1.0 >/dev/null 2>&1
pip install httpx==0.27.2 >/dev/null 2>&1
pip install pydantic==1.10.13 >/dev/null 2>&1

print_status "All Python dependencies installed successfully"
echo

# Step 7: Create .gitignore if it doesn't exist
echo -e "${BLUE}Step 7: Setting up .gitignore...${NC}"
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
*.py[cod]
__pycache__/
.venv/
venv/
ENV/

# Azure Functions
local.settings.json
local.settings*.json
*.local.json
.python_packages/
.func/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Secrets
*.env
.env.*
secrets/
*.key
*.pem
EOF
    print_status "Created .gitignore file"
else
    print_status ".gitignore already exists"
fi
echo

# Step 8: Create a start script
echo -e "${BLUE}Step 8: Creating start script...${NC}"
cat > start.sh << 'EOF'
#!/bin/bash
# Quick start script for the Azure Function

# Activate virtual environment
source .venv/bin/activate

# Start the function
echo "Starting Azure Function on http://localhost:7071"
echo "Press Ctrl+C to stop"
func start
EOF
chmod +x start.sh
print_status "Created start.sh script"
echo

# Final instructions
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✨ Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Update ${YELLOW}local.settings.json${NC} with your Azure credentials"
echo -e "2. Run ${YELLOW}./start.sh${NC} to start the Azure Function"
echo -e "3. In VS Code PORTS tab, make port ${YELLOW}7071${NC} public"
echo -e "4. Update your frontend to use the Codespaces URL"
echo
echo -e "${BLUE}To start the function now, run:${NC}"
echo -e "  ${YELLOW}./start.sh${NC}"
echo
echo -e "${BLUE}Or manually:${NC}"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  ${YELLOW}func start${NC}"
echo

# Ask if user wants to start now
read -p "Do you want to start the Azure Function now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting Azure Function...${NC}"
    ./start.sh
fi