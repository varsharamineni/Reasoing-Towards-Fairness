#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --job-name=reasoning_extract
#SBATCH --output=logs/reasoning_extract_%j.out      # Job stdout (with job ID)
#SBATCH --error=logs/reasoning_extract_%j.err       # Job stderr (with job ID)
#SBATCH --time=02:00:00                              # Max runtime hh:mm:ss
#SBATCH --mem=10G                                    # Memory, adjust as needed
#SBATCH --gres=tmpfs:100G                            # Request 100GB of tmpfs (local SSD /tmp space)
#SBATCH --cpus-per-task=4                            # Adjust CPU cores

# Load your environment if needed (modules, conda, virtualenv, etc.)
module purge
module load python/3.10

echo "🚀 SLURM job started on $(hostname)"
source ~/envs/reasoning_pattern_env/bin/activate

# Use TMPDIR (/tmp) for temporary files — mounted on local SSD
export TMPDIR=/tmp

# Create logs and results directories if they don't exist
mkdir -p logs
mkdir -p results

# Define input and output files
INPUT_FILE="outputs/processed_bbq_checkpoint_results/only_correct/checkpoint_3177/nationality/checkpoint_3177_final.json"
OUTPUT_FILE="results/reasoning_clusters.json"

echo "Running job $SLURM_JOB_ID"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

# Run your Python script with variables
python scripts/extract_reasoning_patterns.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE"

echo "✅ Job finished. Output saved in $OUTPUT_FILE"