#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
dpu-upmem-dpurte-clang -DNR_TASKLETS=16 -DSTACK_SIZE_DEFAULT=256 -O2 -o pim_dpu pim_dpu.c
echo "--------------------------------------------"
echo "Finish Compiling pim_dpu"
echo "--------------------------------------------"

python setup.py develop
echo "--------------------------------------------"
echo "Finish Compiling pim_module"
echo "--------------------------------------------"

ncores=26 #12 #6
nsockets="0"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_emb_pool.py"

#Model param
emb_size=64
emb="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"
rdma_wr_capacity=128
emb_pool_port=1238

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
