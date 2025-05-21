#!/bin/bash

# install.sh - Rebuild and restart Docker containers with fresh dependencies

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