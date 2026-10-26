#!/usr/bin/env bash
# Train the same three experiments with smaller networks, to see how small
# the driving network can be. Nothing existing is overwritten: every run
# writes out/{bc,ppo,search}_h<size>[_s<seed>]_* files.
#
#   source ~/.venvs/odse_policy/bin/activate
#   bash sweep_network_size.sh            # about 30 min on 14 cores
#   python summarize_sweep.py             # table and plot
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/sweep_logs

SIZES=(64,64 32,32 16,16 8,8 16 8 4)
SEEDS=(0 1 2)
PARALLEL=${PARALLEL:-10}

jobs=()
for size in "${SIZES[@]}"; do
    tag=h${size//,/x}
    jobs+=("python train_bc.py --threads 1 --hidden $size --name bc_$tag")
    for seed in "${SEEDS[@]}"; do
        jobs+=("python train_ppo.py --threads 1 --hidden $size --seed $seed --steps 400000 --name ppo_${tag}_s$seed")
        jobs+=("python train_ppo.py --threads 1 --hidden $size --seed $seed --steps 2000000 --scenario mixed --name search_${tag}_s$seed")
    done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
    name=$(echo "{}" | sed "s/.*--name //")
    if [ -f "out/${name}_results.json" ]; then echo "skip $name"; exit 0; fi
    echo "start $name"
    {} > "out/sweep_logs/$name.log" 2>&1 && echo "done  $name" || echo "FAIL  $name"
'
