#!/usr/bin/env bash

# === Required Env Vars === 
# BASETEN_BASE_URL: Base URL for Baseten OpenAI-compatible API (e.g., https://model-xxx.api.baseten.co/production)
# BASETEN_API_KEY: Your Baseten API key (required for authentication)
# MODEL: Model name (e.g., openai/gpt-oss-120b)
# ISL: Input sequence length
# OSL: Output sequence length  
# RANDOM_RANGE_RATIO: Random range ratio for sequence lengths
# CONC: Max concurrency
# RESULT_FILENAME: Output filename for results
# NUM_PROMPTS: Number of prompts to test (defaults to CONC * 10)

# === Optional Env Vars ===
# BASETEN_API_URL: Direct Baseten API endpoint URL (alternative to BASETEN_BASE_URL)

set -e

# Set defaults
NUM_PROMPTS=${NUM_PROMPTS:-$(( ${CONC:-1} * 10 ))}
CONC=${CONC:-1}
ISL=${ISL:-1024}
OSL=${OSL:-1024}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}
MODEL=${MODEL:-"openai/gpt-oss-120b"}
RESULT_FILENAME=${RESULT_FILENAME:-"baseten_gptoss_benchmark.json"}

# Check required environment variables (will check API key later after determining URL)

# Determine the base URL to use
if [ -n "$BASETEN_BASE_URL" ]; then
    BASE_URL="$BASETEN_BASE_URL"
    # Use OpenAI-compatible endpoint
    BACKEND="openai"
elif [ -n "$BASETEN_API_URL" ]; then
    # Use direct Baseten endpoint
    BASE_URL="$BASETEN_API_URL"
    BACKEND="openai"  # Try OpenAI-compatible first
else
    echo "Error: BASETEN_BASE_URL or BASETEN_API_URL must be set"
    exit 1
fi

# Use Baseten API key (required for authentication)
# Unlike other benchmark scripts that test local servers, Baseten is a remote service
# that requires authentication. We'll pass this to the OpenAI client library via
# OPENAI_API_KEY environment variable (standard for OpenAI-compatible APIs)
API_KEY="${BASETEN_API_KEY:-${OPENAI_API_KEY:-}}"

if [ -z "$API_KEY" ]; then
    echo "Error: BASETEN_API_KEY must be set for authentication"
    echo "Get your API key from: https://app.baseten.co/settings/api_keys"
    exit 1
fi

echo "=== Baseten Benchmark Configuration ==="
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"
echo "Backend: $BACKEND"
echo "Input Sequence Length: $ISL"
echo "Output Sequence Length: $OSL"
echo "Concurrency: $CONC"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Result Filename: $RESULT_FILENAME"
echo "======================================="

# Install dependencies if needed
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing datasets..."
    pip install -q datasets pandas
fi

# Clone bench_serving if not present
if [ ! -d "bench_serving" ]; then
    echo "Cloning bench_serving..."
    git clone https://github.com/kimbochen/bench_serving.git
fi

# Export API key for OpenAI client library
# Note: Unlike other scripts that test local servers (no auth needed), Baseten is a remote
# hosted service requiring authentication. The bench_serving tool's 'openai' backend uses
# the OpenAI Python client library, which reads OPENAI_API_KEY from the environment.
# Since Baseten uses OpenAI-compatible APIs, setting OPENAI_API_KEY to your Baseten API
# key is the standard way to authenticate (this is what Baseten's docs recommend).
export OPENAI_API_KEY="$API_KEY"

# Verify API key is set (without exposing the actual key)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set after export!"
    exit 1
fi
echo "✓ API key configured (length: ${#OPENAI_API_KEY} characters)"

# Determine endpoint - use chat/completions for chat models
# If base URL already includes /v1, just append /chat/completions
# Otherwise append /v1/chat/completions
if [[ "$BASE_URL" == */v1 ]]; then
    ENDPOINT="/chat/completions"
elif [[ "$BASE_URL" == */v1/ ]]; then
    ENDPOINT="chat/completions"
else
    ENDPOINT="/v1/chat/completions"
fi

# Use openai-chat backend for chat models (ensures proper endpoint handling)
if [ "$BACKEND" = "openai" ]; then
    BACKEND="openai-chat"
fi

# Set tokenizer (use HuggingFace model name for tokenizer, API model name for API calls)
TOKENIZER="${TOKENIZER:-openai/gpt-oss-120b}"

# Run the benchmark
set -x
python3 bench_serving/benchmark_serving.py \
    --model="$MODEL" \
    --tokenizer="$TOKENIZER" \
    --backend="$BACKEND" \
    --base-url="$BASE_URL" \
    --endpoint="$ENDPOINT" \
    --dataset-name=random \
    --random-input-len="$ISL" \
    --random-output-len="$OSL" \
    --random-range-ratio="$RANDOM_RANGE_RATIO" \
    --num-prompts="$NUM_PROMPTS" \
    --max-concurrency="$CONC" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir="." \
    --result-filename="$RESULT_FILENAME"

echo "Benchmark complete! Results saved to $RESULT_FILENAME"

