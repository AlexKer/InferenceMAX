#!/usr/bin/env bash

# Runner script for Baseten benchmark to match InferenceMAX B200 TRT comparison
# Runs benchmarks at concurrency levels: 4, 8, 16, 32, 64 (matching all B200 TRT test points)
# Usage:
#   export BASETEN_BASE_URL="https://model-xxx.api.baseten.co/production"
#   export BASETEN_API_KEY="your-api-key"
#   ./runners/run_baseten_comparison.sh

set -e

# Default configuration for gptoss120B benchmark (matching InferenceMAX 1k/1k config)
export MODEL=${MODEL:-"openai/gpt-oss-120b"}
export ISL=${ISL:-1024}
export OSL=${OSL:-1024}
export RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}

# Concurrency levels matching all B200 TRT test points from the original run
CONCURRENCIES=(4 8 16 32 64)

# Check for required environment variables
if [ -z "$BASETEN_BASE_URL" ] && [ -z "$BASETEN_API_URL" ]; then
    echo "Error: BASETEN_BASE_URL must be set"
    echo ""
    echo "Usage:"
    echo "  export BASETEN_BASE_URL='https://model-xxx.api.baseten.co/production'"
    echo "  export BASETEN_API_KEY='your-baseten-api-key'"
    echo "  ./runners/run_baseten_comparison.sh"
    echo ""
    echo "Get your Baseten API key from: https://app.baseten.co/settings/api_keys"
    exit 1
fi

if [ -z "$BASETEN_API_KEY" ]; then
    echo "Error: BASETEN_API_KEY must be set"
    echo "Get your Baseten API key from: https://app.baseten.co/settings/api_keys"
    exit 1
fi

# Create a timestamped directory for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="baseten_comparison_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=== Baseten Comparison Benchmark ==="
echo "Model: $MODEL"
echo "Input Length: $ISL"
echo "Output Length: $OSL"
echo "Concurrency Levels: ${CONCURRENCIES[*]}"
echo ""
echo "Note: Delays between runs to respect rate limits (1M TPM)"
echo "  - 30s delay between CONC=4 runs"
echo "  - 60s delay before CONC=8+"
echo "  - 120s delay before CONC=16+"
echo "  - 180s delay before CONC=64 (highest concurrency)"
echo ""
echo "Using NUM_PROMPTS = CONC * 10 (matching InferenceMAX B200 TRT setup)"
echo ""
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run benchmarks at each concurrency level
for i in "${!CONCURRENCIES[@]}"; do
    CONC="${CONCURRENCIES[$i]}"
    
    echo "=========================================="
    echo "Running benchmark with CONC=$CONC"
    echo "=========================================="

    # Add delays to avoid rate limiting, especially for higher concurrency
    if [ $i -gt 0 ]; then
        # Base delay between runs: 30 seconds
        DELAY=30
        
        # Longer delays before high concurrency runs to let rate limits reset
        if [ "$CONC" -ge 64 ]; then
            DELAY=180  # 3 minutes before CONC=64 (highest concurrency)
            echo "Waiting ${DELAY}s before CONC=$CONC to avoid rate limits..."
        elif [ "$CONC" -ge 16 ]; then
            DELAY=120  # 2 minutes before CONC=16+
            echo "Waiting ${DELAY}s before CONC=$CONC to avoid rate limits..."
        elif [ "$CONC" -ge 8 ]; then
            DELAY=60   # 1 minute before CONC=8+
            echo "Waiting ${DELAY}s before CONC=$CONC..."
        else
            echo "Waiting ${DELAY}s before CONC=$CONC..."
        fi
        
        sleep "$DELAY"
    fi

    export CONC=$CONC
    # Match InferenceMAX B200 TRT setup: CONC * 10 prompts for GPT-OSS 120B 1k/1k
    export NUM_PROMPTS=$((CONC * 10))
    export RESULT_FILENAME="${RESULTS_DIR}/baseten_conc${CONC}_${TIMESTAMP}.json"

    cd "$WORKSPACE_DIR"
    bash benchmarks/gptoss_baseten.sh

    if [ $? -ne 0 ]; then
        echo "Error: Benchmark failed for CONC=$CONC. Continuing with other concurrency levels..."
        # Don't exit - continue with other concurrency levels
    fi
    
    echo "Completed CONC=$CONC benchmark"
done

echo ""
echo "=== Benchmark Complete ==="
echo "All results saved to: $RESULTS_DIR/"
echo ""
echo "To compare with B200 TRT results, run:"
echo "  python3 utils/plot_baseten_vs_b200.py $RESULTS_DIR agg_gptoss_1k1k.json"

