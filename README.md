# CloverRec

This is an open source repository for our paper in [OSDI 2026](https://www.usenix.org/conference/osdi26)

> **Enabling Cost-Efficient Recommendation Inference on Real Disaggregated Processing-in-Memory Systems**

## Brief Introduction

Deep recommendation systems enhance user experiences with fast, high-quality personalized recommendations. While memory disaggregation offers a cost-efficient solution for handling large-scale embedding vectors (EVs), it suffers from significant network overheads. Processing-in-memory (PIM) technology alleviates this issue by offloading bandwidth-intensive embedding operations to remote PIM pools, reducing network transfer costs. However, achieving high-performance PIM-enabled embedding remains challenging due to inefficient parallelism, load imbalance, and granularity mismatch. 

To tackle these challenges, we propose CloverRec, a cost-efficient and high-performance recommendation inference system that offloads embedding operations to a disaggregated PIM pool. CloverRec employs dimension parallelism to maximize DPU thread utilization, creates replicas of hot EVs for load balancing across thousands of DPUs, processes DPUs at rank granularity, and pipelines data transfers with DPU execution to mitigate transfer amplification and overhead. Evaluations on a real UPMEM PIM system show that CloverRec outperforms state-of-the-art embedding disaggregation schemes.

## Requirements
- Machines
  - 3 servers: 1 GPU server, 1 host server, and 1 PIM server
- Hardware
  - Mellanox InfiniBand NIC (e.g., ConnectX-5) that supports RDMA
  - Mellanox InfiniBand Switch
  - NVIDIA GPU (e.g., V100) on GPU server
  - UPMEM PIM DIMMs on PIM server
- Software
  - Operating System: Ubuntu 18.04 LTS
  - Programming Language: C++ 11
  - CMake: 3.22 or above
  - Compiler: g++ 11.4.0 or above
  - CUDA: 12.1
  - Python: 3.12
  - Libraries: ibverbs, pyverbs, ldpu

## Build

- Clone this repo
```sh
$ git clone https://anonymous.4open.science/r/CloverRec-E2BE # Anonymize
$ cd CloverRec
```

- Install dependencies
```sh
$ conda create -n CloverRec python=3.12 -y
$ conda activate CloverRec
$ pip install -r requirements.txt
```

## Run

We provide shell scripts for easy running. 

Configure the shell scripts:

(1) Modify the 'server-ip', 'server-port', 'emb-pool-ip', and 'emb-pool-port' to the actual address

(2) Modify the 'raw_file_path' and 'processed_file_path' to the dataset path 

- For the GPU server

```sh
$ cd CloverRec/cloverrec
$ ./benchmark_model_RM1.sh
```

- For the PIM server

```sh
$ cd CloverRec/cloverrec
$ ./benchmark_emb_pool_RM1.sh
```

- For the host server

```sh
$ cd CloverRec/cloverrec
$ ./benchmark_coordinator_RM1.sh {batch_size} {zipf_parameter}
```

