#!/bin/bash

# name argument
if [ $# -lt 1 ]; then
    echo "Provide a name for this task as argument 1."
    exit 1
fi
fileNameAppend=$1

# this script should be in the same directory as runBatchProcessing.slurm
HERE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# For a range
run0=200
runf=300
step=1
for ((run_i = run0; run_i <= runf; run_i += step)); do
    job_name="BEACON_${fileNameAppend}_$run_i"
    echo "Submitting RUN $run_i for processing on job $job_name..."
    sbatch --job-name="$job_name" $HERE/runBatchProcessing.slurm $run_i $fileNameAppend
done


# For a specified list
# runs=(200 230 231)
# for run_i in "${runs[@]}"; do
#     job_name="BEACON_$fileNameAppend_$run_i"
#     echo "Submitting RUN $run_i for processing on job $fileNameAppend..."
#     sbatch --job-name="$job_name" $HERE/runBatchProcessing.slurm $run_i $fileNameAppend
# done
