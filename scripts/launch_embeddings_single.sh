job="knowledge_embeddings"
job_name=$job 
project_path="/lustre/fswork/projects/rech/esq/udg63qz/knowladge/knowledge_processing"
# Check if the tar files path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide the path to a tar file as an argument."
    echo "Usage: $0 <path_to_tar_file>"
    exit 1
fi

if [[ ! "$1" == *.tar ]]; then
    echo "Error: The provided path does not point to a tar file."
    echo "Usage: $0 <path_to_tar_file>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: The specified tar file does not exist."
    echo "Usage: $0 <path_to_tar_file>"
    exit 1
fi
tar_file_path="$1"

# Extract the directory path without the filename
dir_path=$(dirname "$tar_file_path")
# Get the parent directory name
parent_dir=$(basename "$dir_path")
# Construct the output directory path
out_dir="${dir_path%/$parent_dir}/${parent_dir}_embeddings"

echo "Launching job $job_name for single tar file"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=tqh@a100
#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=$project_path/slurm_outputs/%j_out.txt
#SBATCH --error=$project_path/slurm_outputs/%j_err.txt
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread

module purge 
module load arch/a100
module load pytorch-gpu/py3/2.3.0

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

set -x 

srun python $project_path/main.py \
    --mode embedding \
    --tar_file $tar_file_path \
    --output_dir $out_dir \
    --ngpus 1 \
    --batch_size 1024
EOT

