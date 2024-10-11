
out_dir="slurm_outputs"
job="prepocessing"
job_name=$job 
          echo "Launching job $job_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -A tqh@v100
#SBATCH -p prepost
#SBATCH --nodes=1
#SBATCH --output=$out_dir/$job_name/%t_out_%j.txt
#SBATCH --error=$out_dir/$job_name/%t_err_%j.txt
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

module load pytorch-gpu/py3/1.12.1

# Echo des commandes lancees
set -x 
srun python embeddings/scripts/process_wiki_files.py --input outputs/ --output datadir/wikiprocessed
EOT
