job="knowledge_embeddings"
job_name=$job 
project_path="/lustre/fswork/projects/rech/esq/udg63qz/knowladge"
out_dir="slurm_outputs"
tar_files=($project_path/knowledge_processing/datadir/wikiprocessed/data-*-000186.tar)
num_files=${#tar_files[@]}

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
#SBATCH --output=$out_dir/$job_name/%A_%a_out.txt
#SBATCH --error=$out_dir/$job_name/%A_%a_err.txt
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
    --output_dir $project_path/datadir/wikiembeddings/ \
    --ngpus 4 \
    --batch_size 1024
EOT
