#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ncores=26 #12 #6
nsockets="0"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_model.py"

data=random
print_freq=10
rand_seed=727

#Model param
nbatches=500
num_int=119
bot_mlp="2560-512-64"
top_mlp="512-128-1"
emb_size=64
nindices=20
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
interaction="dot"
rpc_type="model"

_args="--num-batches="${nbatches}\
" --data-generation="${data}\
" --num-int="${num_int}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --num-indices-per-lookup-fixed=True"\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --inference-only"\
" --get-cdf-lat=True"\
" --use-gpu"\
" --server-port=8002"


# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - running on 1 GPUs"
echo "--------------------------------------------"

cuda_arg="CUDA_VISIBLE_DEVICES=0"
echo "-------------------"
echo "Using GPUS: 0"
echo "-------------------"

# cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args --use-gpu $dlrm_extra_option > $outf"
cmd="$cuda_arg $dlrm_pt_bin $_args"
echo $cmd
eval $cmd
