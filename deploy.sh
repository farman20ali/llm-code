#!/bin/bash

#############################################
# 🚀 AI-SQL Flask App - One-Click Deploy
#############################################
# This script handles EVERYTHING for deployment
# Author: farman20ali
# Updated: 2025-10-26
#
# 🎯 HOW TO USE:
#
# 1️⃣ EASIEST WAY (Edit defaults, then just press Enter):
#    - Edit DEFAULT_* values below (lines 20-30)
#    - Run: ./deploy.sh
#    - Press Enter for all questions = uses your defaults!
#
# 2️⃣ COPY EXISTING .ENV:
#    - Set EXISTING_ENV_PATH to your .env file location
#    - Run: ./deploy.sh
#    - Script will copy it automatically!
#
# 3️⃣ INTERACTIVE (Type values when asked):
#    - Run: ./deploy.sh
#    - Answer questions (defaults shown in [brackets])
#    - Press Enter to accept default, or type your own
#
#############################################

set -e  # Exit on any error

#############################################
# 📝 CONFIGURATION - EDIT THESE DEFAULTS
#############################################

# Application Settings
APP_NAME="aisql"
REPO_URL="https://github.com/farman20ali/llm-code.git"
APP_DIR="/home/$(whoami)/llm-code"
VENV_DIR="$APP_DIR/venv"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
USER=$(whoami)

# Default Environment Variables (Edit these before running!)
DEFAULT_DB_HOST="0.0.0.0"
DEFAULT_DB_PORT="5432"
DEFAULT_DB_NAME="irs"
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="admin123/?"
DEFAULT_OPENAI_KEY="sk-proj-..."

# Optional: Path to existing .env file (leave empty if you want to create new)
EXISTING_ENV_PATH=""
# Example: EXISTING_ENV_PATH="/home/farman/.env"

#############################################
# Colors for output
#############################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#############################################
# Helper Functions
#############################################

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed!"
        return 1
    fi
    return 0
}

#############################################
# Pre-flight Checks
#############################################

print_step "Pre-flight checks..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script only works on Linux servers"
    exit 1
fi

# Check required commands
check_command git || { print_error "Git is required. Install it first: sudo apt install git"; exit 1; }
check_command python3 || { print_error "Python3 is required. Install it first: sudo apt install python3"; exit 1; }

# Check for pip3, install if missing
if ! check_command pip3; then
    print_warning "pip3 not found, attempting to install..."
    
    # Detect package manager and install pip3
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y python3-pip
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y python3-pip
    else
        print_error "Could not auto-install pip3. Please install it manually."
        exit 1
    fi
    
    # Verify installation
    if check_command pip3; then
        print_success "pip3 installed successfully"
    else
        print_error "Failed to install pip3. Please install manually."
        exit 1
    fi
fi

print_success "All prerequisites satisfied"

#############################################
# Step 1: Clone/Update Repository
#############################################

print_step "Setting up repository..."

if [ -d "$APP_DIR/.git" ]; then
    print_warning "Repository already exists, pulling latest changes..."
    cd "$APP_DIR"
    git pull origin main || { print_error "Git pull failed"; exit 1; }
else
    print_warning "Cloning repository..."
    git clone "$REPO_URL" "$APP_DIR" || { print_error "Git clone failed"; exit 1; }
    cd "$APP_DIR"
fi

print_success "Repository ready at $APP_DIR"

#############################################
# Step 2: Environment Variables Setup
#############################################

print_step "Setting up environment variables..."

if [ ! -f "$APP_DIR/.env" ]; then
    print_warning ".env file not found!"
    
    # Option 1: Copy from existing path
    if [ -n "$EXISTING_ENV_PATH" ] && [ -f "$EXISTING_ENV_PATH" ]; then
        echo -e "\n${YELLOW}Found existing .env at: ${BLUE}$EXISTING_ENV_PATH${NC}"
        read -p "Do you want to copy it? (Y/n): " copy_env
        copy_env=${copy_env:-Y}  # Default to Yes
        
        if [[ $copy_env == "y" || $copy_env == "Y" || $copy_env == "" ]]; then
            cp "$EXISTING_ENV_PATH" "$APP_DIR/.env"
            print_success ".env file copied from $EXISTING_ENV_PATH"
        else
            EXISTING_ENV_PATH=""  # User said no, so create new
        fi
    fi
    
    # Option 2: Ask for path if not set
    if [ -z "$EXISTING_ENV_PATH" ] || [ ! -f "$EXISTING_ENV_PATH" ]; then
        echo -e "\n${YELLOW}Do you have an existing .env file somewhere?${NC}"
        read -p "Enter path to existing .env file (or press Enter to create new): " user_env_path
        
        if [ -n "$user_env_path" ] && [ -f "$user_env_path" ]; then
            cp "$user_env_path" "$APP_DIR/.env"
            print_success ".env file copied from $user_env_path"
        else
            # Option 3: Create new .env with defaults
            echo -e "\n${GREEN}Creating new .env file...${NC}"
            echo -e "${YELLOW}Press Enter to use default values shown in [brackets]${NC}\n"
            
            read -p "Database Host [$DEFAULT_DB_HOST]: " db_host
            db_host=${db_host:-$DEFAULT_DB_HOST}
            
            read -p "Database Port [$DEFAULT_DB_PORT]: " db_port
            db_port=${db_port:-$DEFAULT_DB_PORT}
            
            read -p "Database Name [$DEFAULT_DB_NAME]: " db_name
            db_name=${db_name:-$DEFAULT_DB_NAME}
            
            read -p "Database User [$DEFAULT_DB_USER]: " db_user
            db_user=${db_user:-$DEFAULT_DB_USER}
            
            read -sp "Database Password [$DEFAULT_DB_PASSWORD]: " db_password
            db_password=${db_password:-$DEFAULT_DB_PASSWORD}
            echo
            
            read -p "OpenAI API Key [$DEFAULT_OPENAI_KEY]: " openai_key
            openai_key=${openai_key:-$DEFAULT_OPENAI_KEY}
            
            # Create .env file
            cat > "$APP_DIR/.env" <<EOF
# Database Configuration
DATABASE_URL=postgresql://${db_user}:${db_password}@${db_host}:${db_port}/${db_name}
DB_PROTOCOL=postgresql://
DB_USER=${db_user}
DB_PASSWORD=${db_password}
DB_HOST=${db_host}
DB_PORT=${db_port}
DB_NAME=${db_name}

# OpenAI API Configuration
OPENAI_API_KEY=${openai_key}

# Model Configuration
SCHEMA_FOLDER=scripts
ECONOMY_MODEL=gpt-3.5-turbo
STANDARD_MODEL=gpt-4o-mini
PREMIUM_MODEL=gpt-4o
USE_SCHEMA_AWARE_MODEL=True
SQL_MODEL=ft:gpt-3.5-turbo-0125:personal:accident-sql-generator:BYaKVLaO
COST_TIER=economy

# Application Configuration
PORT=5000
TEMPERATURE=0.0
MAX_TOKENS=2048
EOF
            
            print_success ".env file created with your configuration"
        fi
    fi
else
    print_success ".env file already exists"
    
    # Ask if user wants to update it
    read -p "Do you want to update/recreate .env file? (y/N): " update_env
    update_env=${update_env:-N}  # Default to No
    
    if [[ $update_env == "y" || $update_env == "Y" ]]; then
        rm "$APP_DIR/.env"
        print_warning ".env removed. Re-run this section..."
        # Recursive call to re-enter this section
        bash "$0"
        exit 0
    fi
fi

#############################################
# Step 3: Python Virtual Environment
#############################################

print_step "Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment exists, removing old one..."
    rm -rf "$VENV_DIR"
fi

# Try to create venv, install python3-venv if it fails
if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
    print_warning "python3-venv not installed, installing now..."
    
    # Install python3-venv based on package manager
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-virtualenv
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-virtualenv
    fi
    
    # Try again
python3 -m venv "$VENV_DIR" || { print_error "Failed to create virtual environment"; exit 1; }
fi

source "$VENV_DIR/bin/activate"

print_success "Virtual environment created"

#############################################
# Step 4: Install Dependencies
#############################################

print_step "Installing Python dependencies..."

pip install --upgrade pip
pip install -r requirements.txt || { print_error "Failed to install dependencies"; exit 1; }

# Ensure gunicorn is installed
pip install gunicorn

print_success "All dependencies installed"

#############################################
# Step 5: Database Setup (Optional)
#############################################

print_step "Database initialization..."

if [ -f "$APP_DIR/scripts/init_db.py" ]; then
    read -p "Do you want to initialize/update the database? (y/n): " init_db
    
    if [[ $init_db == "y" || $init_db == "Y" ]]; then
        python "$APP_DIR/scripts/init_db.py" || print_warning "Database init had some issues (may be normal if already initialized)"
        print_success "Database initialization attempted"
    else
        print_warning "Skipping database initialization"
    fi
else
    print_warning "No init_db.py found, skipping database setup"
fi

#############################################
# Step 6: Create Systemd Service
#############################################

print_step "Creating systemd service..."

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=AI-SQL Flask Application
After=network.target

[Service]
User=$USER
Group=www-data
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 --timeout 120 wsgi:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service file created"

#############################################
# Step 7: Start the Service
#############################################

print_step "Starting the application service..."

sudo systemctl daemon-reload
sudo systemctl restart "$APP_NAME"
sudo systemctl enable "$APP_NAME"

sleep 2

# Check service status
if sudo systemctl is-active --quiet "$APP_NAME"; then
    print_success "Service started successfully!"
else
    print_error "Service failed to start"
    echo -e "\n${RED}Service logs:${NC}"
    sudo systemctl status "$APP_NAME" --no-pager -l
    exit 1
fi

#############################################
# Final Summary
#############################################

echo -e "\n${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}    ✅ DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

echo -e "\n${BLUE}📍 Application Details:${NC}"
echo -e "   • Location: ${YELLOW}$APP_DIR${NC}"
echo -e "   • Service: ${YELLOW}$APP_NAME${NC}"
echo -e "   • Port: ${YELLOW}5000${NC}"

echo -e "\n${BLUE}🔧 Useful Commands:${NC}"
echo -e "   • Check status:  ${YELLOW}sudo systemctl status $APP_NAME${NC}"
echo -e "   • View logs:     ${YELLOW}sudo journalctl -u $APP_NAME -f${NC}"
echo -e "   • Restart:       ${YELLOW}sudo systemctl restart $APP_NAME${NC}"
echo -e "   • Stop:          ${YELLOW}sudo systemctl stop $APP_NAME${NC}"

echo -e "\n${BLUE}🌐 Access your app:${NC}"
SERVER_IP=$(hostname -I | awk '{print $1}')
echo -e "   • Local:   ${YELLOW}http://localhost:5000${NC}"
echo -e "   • Network: ${YELLOW}http://$SERVER_IP:5000${NC}"

echo -e "\n${GREEN}═══════════════════════════════════════════${NC}\n"

# Show current service status
sudo systemctl status "$APP_NAME" --no-pager -l
