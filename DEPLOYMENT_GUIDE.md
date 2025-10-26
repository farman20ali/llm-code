# 🚀 Deployment Guide - AI-SQL Flask App

This guide will help you deploy your Flask application to a Linux server in the simplest way possible.

## 📋 Prerequisites

Before you start, make sure you have:

1. **A Linux server** (Ubuntu 20.04+ recommended, or any Linux distro)
2. **SSH access** to your server
3. **Optional**: Your credentials (or just use the defaults in the script!)

## 🎯 Super Easy Deployment

### Option A: Edit Defaults First (Recommended!)

1. **Download the script:**
```bash
curl -sSL https://raw.githubusercontent.com/farman20ali/llm-code/main/deploy.sh -o deploy.sh
```

2. **Edit the defaults at the top:**
```bash
nano deploy.sh
```

Edit these lines (around line 15-25):
```bash
DEFAULT_DB_HOST="0.0.0.0"          # Your database IP
DEFAULT_DB_PORT="5432"
DEFAULT_DB_NAME="irs"
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="admin123/?"
DEFAULT_OPENAI_KEY="sk-proj-..."        # Your OpenAI key

# Optional: Path to existing .env file
EXISTING_ENV_PATH="/home/farman/.env"    # Or leave empty
```

3. **Run it (just press Enter for all questions!):**
```bash
chmod +x deploy.sh
./deploy.sh
# Just press Enter to use your defaults!
```

### Option B: Quick Deploy (Use Defaults)

```bash
curl -sSL https://raw.githubusercontent.com/farman20ali/llm-code/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
./deploy.sh
# Press Enter to use built-in defaults, or type your own values
```

### Option C: Copy Existing .env File

If you already have a `.env` file somewhere:

1. Edit `EXISTING_ENV_PATH` in the script, OR
2. The script will ask you for the path

**That's it!** The script will:
- ✅ Clone your repository
- ✅ Copy existing `.env` OR ask for values (with smart defaults)
- ✅ Set up Python virtual environment
- ✅ Install all dependencies
- ✅ Create systemd service
- ✅ Start your application

---

## 🔐 Environment Variables Explained

The script has **smart defaults** - just press Enter to accept them!

### Database Settings
| Variable | Default Example | Description |
|----------|-----------------|-------------|
| **Database Host** | `0.0.0.0` | IP address or hostname of your PostgreSQL server |
| **Database Port** | `5432` | Port number (default is 5432) |
| **Database Name** | `irs` | Name of your database |
| **Database User** | `postgres` | Database username |
| **Database Password** | `admin123/?` | Database password (input is hidden) |

### OpenAI Settings
| Variable | Default Example | Description |
|----------|-----------------|-------------|
| **OpenAI API Key** | `sk-proj-...` | Your OpenAI API key from platform.openai.com |

### Other Settings (Auto-configured)
These are automatically set in the script:
- `ECONOMY_MODEL=gpt-3.5-turbo`
- `STANDARD_MODEL=gpt-4o-mini`
- `PREMIUM_MODEL=gpt-4o`
- `PORT=5000`
- `COST_TIER=economy`

---

## 📁 Where is Everything Stored?

```
/home/your-username/
└── llm-code/                    # Your app directory
    ├── .env                     # ❗ SECRET - Environment variables (NOT in git)
    ├── .env.example             # ✅ Safe template (in git)
    ├── app.py                   # Main Flask app
    ├── requirements.txt         # Python dependencies
    ├── wsgi.py                  # WSGI entry point
    ├── venv/                    # Python virtual environment
    └── ...
```

### Important Files:

| File/Folder | In Git? | Purpose |
|-------------|---------|---------|
| `.env` | ❌ NO | Contains your actual secrets (database password, API keys) |
| `.env.example` | ✅ YES | Template showing what variables are needed |
| `deploy.sh` | ✅ YES | The deployment script |
| `venv/` | ❌ NO | Python virtual environment (auto-created) |

---

## 🔧 Managing Your Application

### Check if app is running
```bash
sudo systemctl status aisql
```

### View live logs
```bash
sudo journalctl -u aisql -f
```

### Restart the app
```bash
sudo systemctl restart aisql
```

### Stop the app
```bash
sudo systemctl stop aisql
```

### Start the app
```bash
sudo systemctl start aisql
```

### Update to latest code
Just run the deploy script again:
```bash
cd ~/llm-code
./deploy.sh
```

---

## 🌐 Accessing Your Application

After deployment, your app will be available at:

- **From the server itself**: `http://localhost:5000`
- **From other computers**: `http://YOUR_SERVER_IP:5000`

### Test if it's working:
```bash
curl http://localhost:5000
```

---

## ❓ Troubleshooting

### Problem: Service won't start

**Solution**: Check the logs
```bash
sudo journalctl -u aisql -n 50 --no-pager
```

Common issues:
- Wrong database credentials in `.env`
- Database server not accessible
- Missing Python dependencies

### Problem: Can't access from external network

**Solution**: Open port 5000 in firewall

For Ubuntu/Debian:
```bash
sudo ufw allow 5000
sudo ufw enable
```

For CentOS/RHEL:
```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

### Problem: Need to change environment variables

**Solution**: Edit the .env file
```bash
nano ~/llm-code/.env
# Make your changes, then save (Ctrl+X, Y, Enter)

# Restart the service
sudo systemctl restart aisql
```

---

## 🔄 Re-deploying / Updating

Whenever you push new code to GitHub:

1. SSH into your server
2. Run the deploy script again:
```bash
cd ~/llm-code
./deploy.sh
```

The script is smart enough to:
- Pull latest code
- Keep your existing `.env` file
- Update dependencies if needed
- Restart the service

---

## 🆘 Quick Reference Commands

```bash
# Deploy/Update app
cd ~/llm-code && ./deploy.sh

# Service management
sudo systemctl status aisql      # Check status
sudo systemctl restart aisql     # Restart
sudo systemctl stop aisql        # Stop
sudo systemctl start aisql       # Start

# View logs
sudo journalctl -u aisql -f      # Live logs
sudo journalctl -u aisql -n 100  # Last 100 lines

# Edit configuration
nano ~/llm-code/.env             # Edit environment variables

# Test the API
curl http://localhost:5000       # Basic test
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'      # Test API endpoint
```

---

## 🔒 Security Best Practices

1. **Never commit `.env` to git** - It's already in `.gitignore`
2. **Use strong database passwords**
3. **Keep your OpenAI API key secret**
4. **Consider using a reverse proxy** (nginx) for production
5. **Enable firewall** and only open necessary ports
6. **Regular updates**: Run deploy script to get latest code

---

## 📞 Need Help?

If you encounter issues:

1. Check the logs: `sudo journalctl -u aisql -f`
2. Verify `.env` file has correct values
3. Make sure database is accessible
4. Test OpenAI API key at platform.openai.com

---

**Happy Deploying! 🎉**
