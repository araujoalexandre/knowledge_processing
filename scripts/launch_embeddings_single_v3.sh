job="knowledge_embeddings"
job_name=$job 
project_path="/lustre/fswork/projects/rech/esq/udg63qz/knowladge/knowledge_processing"
out_dir="slurm_outputs"
tar_file="$project_path/datadir/wikiprocessed/data-000001-000186.tar"

echo "Launching job $job_name for single tar file"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=tqh@a100
#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=$out_dir/$job_name/%j_out.txt
#SBATCH --error=$out_dir/$job_name/%j_err.txt
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread

module purge 
module load cpuarch/amd
module load pytorch-gpu/py3/2.3.0

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# Echo des commandes lancees
set -x 

srun python $project_path/main.py \
    --mode embedding \
    --tar_file $tar_file \
    --output_dir $project_path/datadir/wikiembeddings/ \
    --ngpus 4 \
    --batch_size 1024
EOT
