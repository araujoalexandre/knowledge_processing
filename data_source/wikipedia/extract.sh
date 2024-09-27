#!/bin/bash
#SBATCH --job-name=wikiextract_job    # Job name
#SBATCH --output=wikiextract_output.txt  # Output file
#SBATCH --error=wikiextract_error.txt   # Error file
#SBATCH --partition=prepost             # Partition name
#SBATCH --ntasks=1                      # Run on a single task
#SBATCH --cpus-per-task=10               # Number of CPU cores per task
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --account=esq@v100

# Load any required modules (optional)
# module load python/3.8

# Run your command
python -m wikiextractor.WikiExtractor datadir/enwiki-latest-pages-articles.xml.bz2 --processes 10 --json -o outputs/


