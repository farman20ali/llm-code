#!/bin/bash

# CONFIGURATION
APP_NAME="aisql"
APP_DIR="/home/farman/farman_ws/llm-code"
REPO_URL="https://github.com/your-username/your-repo.git"   # <-- Change this
ENV_SOURCE_PATH="/home/farman/.env"                         # <-- Where the .env lives now
VENV_DIR="$APP_DIR/venv"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
USER=$(whoami)

echo "🚀 Starting Flask app deployment..."

# Step 1: Clone or pull the Git repository
if [ -d "$APP_DIR/.git" ]; then
    echo "[*] Git repo found, pulling latest changes..."
    cd "$APP_DIR" || exit 1
    git pull origin main
else
    echo "[*] Cloning Git repo..."
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR" || exit 1
fi

# Step 2: Copy the .env file
echo "[*] Copying .env file..."
cp "$ENV_SOURCE_PATH" "$APP_DIR/.env"

# Step 3: Set up virtual environment
echo "[*] Setting up virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Step 4: Install dependencies
echo "[*] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Install gunicorn if missing
if ! command -v gunicorn &> /dev/null; then
    echo "[*] Installing gunicorn..."
    pip install gunicorn
fi

# Step 6: Create systemd service file
echo "[*] Writing systemd service to $SERVICE_FILE..."

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Gunicorn instance to serve AI SQL Flask App
After=network.target

[Service]
User=$USER
Group=www-data
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 wsgi:app

[Install]
WantedBy=multi-user.target
EOF

# Step 7: Reload systemd and start the service
echo "[*] Starting systemd service..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart "$APP_NAME"
sudo systemctl enable "$APP_NAME"

echo "✅ Deployment complete. Service status:"
sudo systemctl status "$APP_NAME" --no-pager
