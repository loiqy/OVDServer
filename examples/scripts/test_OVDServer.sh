#!/bin/bash

# Check if backend parameter is provided
if [ -z "$2" ]; then
    echo "Usage: $0 <backend> $1 <model_name>"
    exit 1
fi

backend=$1
model_name=$2

python -m test_OVDServer --backend "$backend" --model_name "$model_name" --host 127.0.0.1 --port 8081