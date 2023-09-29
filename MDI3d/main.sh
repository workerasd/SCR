#!/usr/bin/env bash
gpu=0
python main.py --gpu_id $gpu --tgt rc --batch 36 --name svd-F2 --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt rc --batch 36 --name svd-F2 --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt rc --batch 36 --name svd-F2 --lr 0.01 --seed 2 &

python main.py --gpu_id 1 --tgt rl --batch 36 --name svd-F2 --lr 0.01 --seed 0 &
python main.py --gpu_id 1 --tgt rl --batch 36 --name svd-F2 --lr 0.01 --seed 1 &
python main.py --gpu_id 1 --tgt rl --batch 36 --name svd-F2 --lr 0.01 --seed 2

python main.py --gpu_id $gpu --tgt t --batch 36 --name svd-F2 --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt t --batch 36 --name svd-F2 --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt t --batch 36 --name svd-F2 --lr 0.01 --seed 2