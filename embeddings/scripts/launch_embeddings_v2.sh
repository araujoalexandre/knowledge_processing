job="knowledge_embeddings"
job_name=$job 
out_dir="slurm_outputs"
          echo "Launching job $job_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -A tqh@a100
#SBATCH -p gpu_p5
#SBATCH --gres=gpu:6
#SBATCH -C a100
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --output=$out_dir/$job_name/%t_out.txt
#SBATCH --error=$out_dir/$job_name/%t_err.txt
#SBATCH --time=05:00:00
#SBATCH --hint=nomultithread

module purge 
module load cpuarch/amd
module load pytorch-gpu/py3/2.3.0

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# Echo des commandes lancees
set -x 
srun python embeddings/create_embeddings_distributed_v2.py --input_dir datadir/wikiprocessed --output_dir datadir/processed_embed_v2 --batch_size 1536
EOT
