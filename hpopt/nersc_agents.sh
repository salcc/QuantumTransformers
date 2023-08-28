#!/bin/bash

read -p "Are you sure you want to submit ${3} jobs? ([y]es or [N]o) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    for _ in $(seq "${3}")
    do
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="${1}"
#SBATCH -Am4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --output=/global/homes/s/salcc/QuantumTransformers/hpopt/logs/slurm-%x-%j.out

export SLURM_CPU_BIND="cores"
cd /global/homes/s/salcc/QuantumTransformers/hpopt
python agent.py "${1}" "${2}"
EOT
    done

    sleep 5
    squeue --me -o "%.18i %.12P %.40j %.8u %.2t %.10M %.6D %R"
fi    
