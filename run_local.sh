#!/bin/bash
# Helper script to run DVC/MLflow commands locally with proper environment variables

# Load environment variables from .env.local
if [ -f .env.local ]; then
    echo "Loading environment variables from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
    echo "✓ Environment configured for local development"
else
    echo "⚠️  .env.local not found. Creating it..."
    cat > .env.local <<EOF
# Local environment overrides
# This file is for running DVC commands on your host machine
# It's gitignored and won't be committed

# For local host machine (outside Docker)
AWS_ENDPOINT_URL_S3=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_DEFAULT_REGION=us-east-1
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
EOF
    echo "✓ Created .env.local"
    export $(grep -v '^#' .env.local | xargs)
fi

# Run the command passed as arguments
echo "Running: $@"
exec "$@"
