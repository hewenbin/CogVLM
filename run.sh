#!/bin/bash

conda activate cogvlm

CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 0 >> gpu0.stdout 2>> gpu0.stderr &
CUDA_VISIBLE_DEVICES=1 taskset -c 8-15 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 1 >> gpu1.stdout 2>> gpu1.stderr &
CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 2 >> gpu2.stdout 2>> gpu2.stderr &
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 3 >> gpu3.stdout 2>> gpu3.stderr &
CUDA_VISIBLE_DEVICES=4 taskset -c 32-39 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 4 >> gpu4.stdout 2>> gpu4.stderr &
CUDA_VISIBLE_DEVICES=5 taskset -c 40-47 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 5 >> gpu5.stdout 2>> gpu5.stderr &
CUDA_VISIBLE_DEVICES=6 taskset -c 48-55 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 6 >> gpu6.stdout 2>> gpu6.stderr &
CUDA_VISIBLE_DEVICES=7 taskset -c 56-63 python cli_demo.py --from_pretrained cogvlm-chat --top_p 0.4 --top_k 5 --temperature 0.8 --version chat --english --bf16 --n-workers 8 --worker-idx 7 >> gpu7.stdout 2>> gpu7.stderr &
wait
