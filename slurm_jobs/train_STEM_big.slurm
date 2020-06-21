#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128000
#SBATCH --job-name="train_STEM_big"
#SBATCH --output=train_STEM_big.out
#SBATCH --mail-user=camillmd@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
  
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module load GCCcore/.8.3.0
module load Python/3.7.4
source ../../.virtualenvs/masters-project/bin/activate

cd ../code
python pretrain_DAMSM.py --data_size big --cfg cfg/pretrain_STEM.yml

