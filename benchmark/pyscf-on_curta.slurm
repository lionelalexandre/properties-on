#!/bin/bash
#SBATCH -o pyscf-on.%j.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --constraint=compute
#SBATCH -t 00-00:10:00
set -x

# Load modules, unless already done before job is submitted (remember,
# SLURM export the environment as per default, unless you 
# set --export=NONE

# load modules
module purge
module load use.own
module load pyscf-on/pyscf-on-mkl

# python input file (.py) and molecules xyz list
python_input='benchmark.py'
molecules_xyz_list='benchmark.list'
molecules_xyz_dir='xyz'

# create a job ID
export SJID="$SLURM_JOB_NAME"_"`echo $SLURM_JOB_ID | cut -d . -f 1`_`date '+%d%m%Y'`"

# create job directory in scratch 
SCRDIR=/scratch/$USER/$SJID
mkdir -p $SCRDIR

# define submission file directory
D0=$SLURM_SUBMIT_DIR

# copy input files to the scratch directory
for file in $python_input $molecules_xyz_list #$(cat $molecule_xyz_list) 
do
  cp $file $SCRDIR/
done

# copy xyz files directory
cp -R $molecules_xyz_dir $SCRDIR/

# move to scratch directory 
cd $SCRDIR

# execute the python script
python3.9 $python_input  > output.dat

# create results directory 
mkdir $SLURM_SUBMIT_DIR/$SJID

# we copy results to home 
cp -R * $SLURM_SUBMIT_DIR/$SJID
