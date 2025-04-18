#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(commit_id, configpath, nruns):
    return f"""#!/bin/bash

#SBATCH --job-name=pfe_sondra
#SBATCH --nodes=1
#SBATCH --exclude=sh00,sh[10-19]
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "gpu info :"
nvidia-smi

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

export WANDB_API_KEY="58de5b94da8d662be3569ed9671373d4e582ab2c" # added wandb access token 

# Install the project dependencies
python -m pip install --upgrade pip setuptools && python -m pip install -r piprequirements.txt


echo "Training"
python -m src.torchtmpl.main {configpath} train visualize

echo "Saving logs/checkpoints from $TMPDIR back to $current_dir"
rsync -r $TMPDIR/code/logs/ $current_dir/logs/ 2>/dev/null || true
rsync -r $TMPDIR/code/checkpoints/ $current_dir/checkpoints/ 2>/dev/null || true

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [2, 3]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_configfilepath, nruns))
