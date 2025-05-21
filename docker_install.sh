#!/bin/bash

# install.sh - Sync code and rebuild Docker environment inside llm-code project

REPO_URL="https://github.com/farman20ali/llm-code"
PROJECT_DIR="llm-code"

# Function to run docker commands from project root
run_docker_commands() {
  echo "Stopping and removing existing containers..."
  docker-compose down

  echo "Pruning unused Docker data..."
  docker system prune -f

  echo "Rebuilding Docker containers without cache..."
  docker-compose build --no-cache

  echo "Starting Docker containers..."
  docker-compose up -d

  echo "Fetching logs for 'web' container..."
  docker logs $(docker ps -q --filter name=web) | cat

  echo "Done!"
}

# Get current directory name
CURRENT_DIR_NAME=${PWD##*/}

if [ "$CURRENT_DIR_NAME" == "$PROJECT_DIR" ]; then
  echo "Already inside project directory. Pulling latest changes..."
  git pull || { echo "❌ Git pull failed."; exit 1; }
  run_docker_commands
elif [ -d "$PROJECT_DIR" ]; then
  echo "Project directory found. Entering and pulling latest changes..."
  cd "$PROJECT_DIR" || { echo "❌ Failed to cd into project directory."; exit 1; }
  git pull || { echo "❌ Git pull failed."; exit 1; }
  run_docker_commands
else
  echo "Project directory not found. Cloning repository..."
  git clone "$REPO_URL" || { echo "❌ Git clone failed."; exit 1; }
  cd "$PROJECT_DIR" || { echo "❌ Failed to cd into cloned project."; exit 1; }
  run_docker_commands
fi