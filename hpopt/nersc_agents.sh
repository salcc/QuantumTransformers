#!/bin/bash

for _ in $(seq "${3}")
do
    sbatch <<EOT
#!/bin/bash
#SBATCH -Am4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --output=/global/homes/s/salcc/QuantumTransformers/hpopt/logs/slurm-"${1}"-%j.out

export SLURM_CPU_BIND="cores"
cd /global/homes/s/salcc/QuantumTransformers/hpopt
python agent.py "${1}" "${2}"
EOT
done

sleep 5
squeue --me
