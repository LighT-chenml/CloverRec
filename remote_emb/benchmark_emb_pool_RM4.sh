#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
ncores=26 #12 #6
nsockets="0"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_emb_pool.py"

#Model param
emb_size=128
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
# emb="10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000"
rdma_wr_capacity=128
emb_pool_port=1237

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - running on 1 GPUs"
echo "--------------------------------------------"

cuda_arg="CUDA_VISIBLE_DEVICES=0"
echo "-------------------"
echo "Using GPUS: 0"
echo "-------------------"

# cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args --use-gpu $dlrm_extra_option > $outf"
cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=${emb_size} --arch-embedding-size=${emb} --rdma-wr-capacity=${rdma_wr_capacity} --emb-pool-port=${emb_pool_port}"
echo $cmd
eval $cmd
