job="knowledge_embeddings"
job_name=$job 
project_path="/lustre/fswork/projects/rech/esq/udg63qz/knowladge"
# Check if the tar files path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide the path to tar files as an argument."
    echo "Usage: $0 <path_to_tar_files>"
    exit 1
fi
tar_files_path="$1"
tar_files=($tar_files_path/data*.tar)
num_files=${#tar_files[@]}
out_dir="${tar_files_path}_embeddings"

echo "Launching job array $job_name with $num_files tasks"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=tqh@a100
#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:6
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=$project_path/knowledge_processing/slurm_outputs/$job_name/%A_%a_out.txt
#SBATCH --error=$project_path/knowledge_processing/slurm_outputs/$job_name/%A_%a_err.txt
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-$((num_files - 1))

module purge 
module load cpuarch/amd
module load pytorch-gpu/py3/2.3.0

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# Echo des commandes lancees
set -x 

# Get the current tar file based on the array task ID
current_tar_file=\${tar_files[\$SLURM_ARRAY_TASK_ID]}

srun python $project_path/main.py \
    --mode embedding \
    --tar_file $current_tar_file \
    --output_dir $out_dir \
    --ngpus 4 \
    --batch_size 1024
EOT
