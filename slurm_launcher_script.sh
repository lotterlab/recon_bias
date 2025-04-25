#!/usr/bin/bash

#SBATCH -J "mri_inr"   # job name
#SBATCH --time=5-00:00:00   # walltime
#SBATCH --output=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.err  # Standard error of the script
#SBATCH --mem=64G
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:0

# load python module
. "/opt/anaconda3/etc/profile.d/conda.sh"

# activate corresponding environment
conda deactivate
conda activate dfci

cd "/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/code/recon_bias"

python3 src/scripts/generate_low_dose_dataset.py --source_dir /vol/miltank/datasets/CheXpert --target_dir /vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/data/CheXpert_noisy --metadata /vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/data/CheXpert/chex-metadata.csv --photon_counts "3e3,1e4,1e5"