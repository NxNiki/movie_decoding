#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o ../../../brain_decoding-output/joblog_$JOB_ID.txt
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=2:00:00,h_data=100G,highp,gpu=1
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
# #$ -M $USER@mail
# Notify when
# #$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
# To see which versions of anaconda are available use: module av anaconda
module load anaconda3
echo "loaded anaconda"

# activate an already existing conda environment (CHANGE THE NAME OF THE ENVIRONMENT):
conda activate movie_decoding

# in the following two lines substitute the command with the
# needed command below:
python main.py

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
### extract_clusterless_parallel.job STOP ###
# this site shows how to do array jobs: https://info.hpc.sussex.ac.uk/hpc-guide/how-to/array.html
# (better than the Hoffman site https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing.html#how-to-build-a-submission-script)
