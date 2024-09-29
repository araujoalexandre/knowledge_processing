
out_dir="slurm_outputs"
job="knowledge_embeddings"
job_name=$job 
          echo "Launching job $job_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -A tqh@a100
#SBATCH -p gpu_p5
#SBATCH -C a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=$out_dir/$job_name/%t_out_%j.txt
#SBATCH --error=$out_dir/$job_name/%t_err_%j.txt
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

module purge 
module load cpuarch/amd
module load pytorch-gpu/py3/1.12.1

# Echo des commandes lancees
set -x 
srun python embeddings/create_embeddings_distributed.py --path_to_files datadir/wikiprocessed --out datadir/wikiembeddings --batch_size 1024
EOT
