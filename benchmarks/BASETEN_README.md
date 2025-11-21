# Baseten Inference Benchmark

This directory contains scripts for benchmarking Baseten inference endpoints, specifically designed to test end-to-end latency for models like GPT-OSS-120B.

## Quick Start

### Prerequisites

1. A Baseten API endpoint URL (either direct endpoint or OpenAI-compatible endpoint)
2. An API key (if required)
3. Python 3 with `datasets` and `pandas` installed

### Running the Benchmark

#### Option 1: Using the Runner Script (Recommended)

```bash
# Set your Baseten endpoint and API key
export BASETEN_BASE_URL="https://model-xxx.api.baseten.co/production"
export BASETEN_API_KEY="your-baseten-api-key-here"

# Run with defaults (GPT-OSS-120B, 1024 input, 1024 output, concurrency 4)
./runners/run_baseten_benchmark.sh
```

#### Option 2: Direct Script Execution

```bash
# Set environment variables
export BASETEN_BASE_URL="https://model-xxx.api.baseten.co/production"
export BASETEN_API_KEY="your-baseten-api-key-here"
export MODEL="openai/gpt-oss-120b"
export ISL=1024
export OSL=1024
export CONC=4
export RESULT_FILENAME="my_benchmark_results.json"

# Run the benchmark
bash benchmarks/gptoss_baseten.sh
```

**Note:** You need a Baseten API key, not an OpenAI API key. Get your API key from your [Baseten account settings](https://app.baseten.co/settings/api_keys).

## Configuration Options

### Environment Variables

- **BASETEN_BASE_URL** (required): Base URL for Baseten OpenAI-compatible API endpoint
  - Example: `https://model-xxx.api.baseten.co/production`
  - The script will append `/v1/chat/completions` automatically

- **BASETEN_API_URL** (alternative): Direct Baseten API endpoint URL
  - Use this if you have a direct endpoint URL instead of OpenAI-compatible

- **BASETEN_API_KEY** (required): Your Baseten API key for authentication
  - Get your API key from: https://app.baseten.co/settings/api_keys
  - Note: This is your Baseten API key, not an OpenAI API key. The script uses it with the OpenAI-compatible client library.

- **MODEL**: Model identifier (default: `openai/gpt-oss-120b`)

- **ISL**: Input sequence length (default: `1024`)

- **OSL**: Output sequence length (default: `1024`)

- **CONC**: Maximum concurrency (default: `4`)

- **NUM_PROMPTS**: Number of prompts to test (default: `CONC * 10`)

- **RANDOM_RANGE_RATIO**: Random range ratio for sequence lengths (default: `0.0`)

- **RESULT_FILENAME**: Output filename for results (default: auto-generated with timestamp)

## Understanding Results

The benchmark measures the following metrics:

- **ttft**: Time to First Token - latency until the first token is generated
- **tpot**: Time per Output Token - average time per token after the first token
- **itl**: Inter-Token Latency - latency between tokens
- **e2el**: End-to-End Latency - total latency for the complete response

Results are saved as JSON with percentile statistics (p50, p90, p95, p99).

## Example Output

After running the benchmark, you'll get a JSON file with results like:

```json
{
  "ttft": {
    "p50": 0.234,
    "p90": 0.456,
    "p95": 0.567,
    "p99": 0.789
  },
  "e2el": {
    "p50": 2.345,
    "p90": 3.456,
    "p95": 4.567,
    "p99": 5.678
  },
  ...
}
```

## Customizing the Benchmark

### Different Sequence Lengths

```bash
export ISL=2048
export OSL=512
./runners/run_baseten_benchmark.sh
```

### Higher Concurrency

```bash
export CONC=16
export NUM_PROMPTS=160  # CONC * 10
./runners/run_baseten_benchmark.sh
```

### Different Models

```bash
export MODEL="other/model-name"
./runners/run_baseten_benchmark.sh
```

## Troubleshooting

### Authentication Issues

If you get authentication errors, make sure:
- Your Baseten API key is correctly set (use `BASETEN_API_KEY`, not `OPENAI_API_KEY`)
- The API key has the right permissions
- You're using the correct endpoint URL
- Get your API key from: https://app.baseten.co/settings/api_keys

### Connection Issues

- Verify the endpoint URL is correct
- Check if your network can reach the Baseten endpoint
- Ensure the endpoint is active and not rate-limited

### Script Errors

- Make sure `bench_serving` repository is cloned (script does this automatically)
- Ensure Python dependencies are installed: `pip install datasets pandas`
- Check that the endpoint supports OpenAI-compatible API format

## Notes

- The benchmark uses the `bench_serving` tool from https://github.com/kimbochen/bench_serving.git
- Baseten endpoints should support OpenAI-compatible API format (`/v1/chat/completions`)
- The benchmark sends concurrent requests to measure real-world performance
- Results include percentile metrics to understand latency distribution

