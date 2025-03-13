#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
ncores=26 #12 #6
nsockets="0"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_coordinator.py"

data=random
print_freq=10
rand_seed=727

#Model param
batch_size=$1
nbatches=1000
bot_mlp="256-128-64"
top_mlp="256-64-1"
emb_size=64
nindices=80
# emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
emb="100000-100000-100000-100000-100000-100000-100000-100000-100000-100000"
# emb="10000-10000-10000-10000-10000-10000-10000-10000-10000-10000"
interaction="dot"
rpc_type="coordinator"

_args="--num-batches="${nbatches}\
" --data-generation="${data}\
" --rand-data-dist=zipfian"\
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
" --server-ip=10.0.0.5"\
" --server-port=8000"\
" --emb-pool-ip=10.0.0.11"\
" --emb-pool-port=1234"\
" --zipf-parameter=1.5"

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - running on 1 GPUs"
echo "--------------------------------------------"

cuda_arg="CUDA_VISIBLE_DEVICES=0"
echo "-------------------"
echo "Using GPUS: 0"
echo "Batch Size: "$batch_size
echo "-------------------"

# cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args --use-gpu $dlrm_extra_option > $outf"
cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size $batch_size $_args"
echo $cmd
eval $cmd
