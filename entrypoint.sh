#!/bin/bash
set -e

echo "🔧 Fixing permissions for workspace..."

# Fix permissions for .dvc directory
if [ -d "/home/airflow/workspace/.dvc" ]; then
    chmod -R 777 /home/airflow/workspace/.dvc 2>/dev/null || true
    echo "✓ .dvc permissions fixed"
fi

# Fix permissions for data directory
if [ -d "/home/airflow/workspace/data" ]; then
    chmod -R 777 /home/airflow/workspace/data 2>/dev/null || true
    echo "✓ data permissions fixed"
fi

# Fix permissions for logs
if [ -d "/home/airflow/workspace/logs" ]; then
    chmod -R 777 /home/airflow/workspace/logs 2>/dev/null || true
    echo "✓ logs permissions fixed"
fi

echo "✅ Permissions setup complete"

# DVC will automatically use AWS_ENDPOINT_URL_S3 environment variable
# No need to create config.local anymore

# Execute the original command with airflow prefix if needed
if [ "$1" = "webserver" ] || [ "$1" = "scheduler" ] || [ "$1" = "worker" ] || [ "$1" = "celery" ]; then
    exec airflow "$@"
else
    exec "$@"
fi
