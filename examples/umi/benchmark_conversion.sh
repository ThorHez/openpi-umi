#!/bin/bash
# Benchmark script to compare conversion speeds
# Usage: ./benchmark_conversion.sh <input_zarr_file>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_zarr_file>"
    echo "Example: $0 dataset.zarr.zip"
    exit 1
fi

INPUT_FILE="$1"
BASE_NAME=$(basename "$INPUT_FILE" .zarr.zip)
REPO_ID="benchmark/${BASE_NAME}"

echo "=========================================="
echo "Conversion Speed Benchmark"
echo "=========================================="
echo ""
echo "Input: $INPUT_FILE"
echo ""

# Function to time execution
time_conversion() {
    local script=$1
    local output_dir=$2
    local extra_args=$3
    
    echo "Testing: $script"
    echo "Output: $output_dir"
    echo "Starting at: $(date)"
    
    # Clean output directory if exists
    rm -rf "$output_dir"
    
    # Time the conversion
    START=$(date +%s)
    python "$script" \
        --input "$INPUT_FILE" \
        --output "$output_dir" \
        --repo-id "$REPO_ID" \
        --fps 30 \
        $extra_args
    END=$(date +%s)
    
    DURATION=$((END - START))
    echo "✓ Completed in: ${DURATION}s"
    echo ""
    
    # Clean up
    rm -rf "$output_dir"
}

echo "----------------------------------------"
echo "1. Original (Single Process)"
echo "----------------------------------------"
time_conversion \
    "convert_umi_data_to_lerobot.py" \
    "./benchmark_original" \
    ""

echo "----------------------------------------"
echo "2. Parallel (8 workers, batch 100)"
echo "----------------------------------------"
time_conversion \
    "convert_umi_data_to_lerobot_parallel.py" \
    "./benchmark_parallel" \
    "--workers 8 --batch-size 100"

echo "----------------------------------------"
echo "3. Optimized Fast (8 workers)"
echo "----------------------------------------"
time_conversion \
    "convert_umi_data_to_lerobot_fast.py" \
    "./benchmark_fast" \
    "--workers 8 --load-batch-size 50 --process-batch-size 10"

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="

