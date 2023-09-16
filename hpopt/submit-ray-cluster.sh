#!/bin/bash

read -p "Are you sure you want to run ${2} tasks? ([y]es or [N]o) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Script based on https://github.com/NERSC/slurm-ray-cluster/blob/master/submit-ray-cluster.sbatch
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="${1}"
#SBATCH --output=/global/homes/s/salcc/QuantumTransformers/hpopt/logs/slurm-%x-%j.out

#SBATCH -Am4392
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=${2}

### Give all resources on each node to a single Ray task, Ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32

cd /global/homes/s/salcc/QuantumTransformers/hpopt

head_node=\$(hostname)
head_node_ip=\$(hostname --ip-address)
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "\$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"\$head_node_ip"
if [[ \${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=\${ADDR[1]}
else
  head_node_ip=\${ADDR[0]}
fi
fi
port=6379

echo "STARTING HEAD at \$head_node"
echo "Head node IP: \$head_node_ip"
srun --nodes=1 --ntasks=1 -w \$head_node start-head.sh \$head_node_ip &
sleep 10

worker_num=\$((\$SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
srun -n \$worker_num --nodes=\$worker_num --ntasks-per-node=1 --exclude \$head_node start-worker.sh \$head_node_ip:\$port &
sleep 5
##############################################################################################

#### call your code below
python hpopt.py "${1}" --trials "${2}" "${@:3}"

exit
EOT
    sleep 10
    squeue --me -o "%.18i %.12P %.40j %.8u %.2t %.10M %.6D %R"
fi
