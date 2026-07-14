# CloverRec

This repository contains the artifact for CloverRec:

> **CloverRec: Cost-Efficient Disaggregated Processing-in-Memory Systems for Deep Recommendation Inference**

Deep recommendation systems enhance user experiences by providing fast, high-quality personalized recommendations, which requires large-capacity and high-speed memory. While memory disaggregation offers a cost-efficient solution for handling large-scale embedding vectors (EVs), it suffers from significant network overheads. Processing-in-memory (PIM) technology restructures the data access path by offloading bandwidth-intensive embedding operations to remote PIM pools, hence minimizing data movement and alleviating network transfer costs. However, fully exploiting the bandwidth potential of disaggregated PIM pools is hindered by sub-optimal intra-unit parallelism, cross-unit load imbalance, and granularity mismatch in the host-device data transfer. To tackle these challenges, we propose CloverRec, a cost-efficient and high-performance recommendation inference system that offloads embedding operations to a disaggregated PIM pool. CloverRec employs a fine-grained dimension-partitioned data layout to achieve synchronization-free parallelism, introduces a dynamic data placement mechanism to mitigate hot-spot skewness across thousands of PIM units, and orchestrates hierarchy-aware data transfer at rank granularity to suppress transfer amplification and saturate the host-device bandwidth. Evaluations on a real UPMEM PIM system show that CloverRec outperforms state-of-the-art embedding schemes.

## Artifact Scope

The repository includes four systems:

- `cloverrec`: CloverRec with PIM embedding execution, remote embedding transfer,
  client-side cache, and adaptive hot-embedding replication.
- `naive_pim_emb`: a naive PIM embedding baseline.
- `remote_emb`: a remote embedding baseline without PIM compute.
- `local_emb`: a local CPU embedding baseline.

The preferred entry points for new runs are the scripts under `scripts/`.
The `scripts/run_smoke.sh` wrapper forwards to the Python launcher.
The scripted workloads include synthetic `RM1`-`RM4` runs and an optional
`KAGGLE` workload for the processed Kaggle/Criteo data path.

## Server Access for Evaluators

The artifact is evaluated on our preconfigured lab servers. Evaluators connect
to the PM2 GPU server through Tailscale and authenticate with an SSH public key.

1. Install Tailscale from <https://tailscale.com/download> and sign in.
2. Open the [CloverRec Tailscale invitation link](https://login.tailscale.com/admin/invite/poqxLcZrEwiwuJxzh6uT11)
   and accept the invitation. If the link has expired, contact the paper authors
   for a new invitation.
3. Send your **SSH public key** (for example, `~/.ssh/id_ed25519.pub`) to the
   authors. If you do not have one, create it with `ssh-keygen -t ed25519`. The
   authors will install the public key for user `cml` on PM2. Never send your
   private key.
4. After the authors confirm that the key has been installed, verify that PM2 is
   visible with `tailscale status`, then connect with:

```sh
ssh cml@100.123.137.1
cd /home/cml/CloverRec
```

Example VS Code Remote SSH configuration:

```sshconfig
Host CloverRec-PM2
    HostName 100.123.137.1
    User cml
    Port 22
```

Only PM2 needs to be reached through Tailscale. The experiment launcher runs on
PM2 and starts processes on the PIM server over the lab's internal network. In
the commands below, `192.168.123.*` addresses are internal management/SSH
addresses, while `10.0.0.*` addresses are InfiniBand/RDMA data-plane addresses.
For example, the default run uses `--model-ip 10.0.0.5`,
`--emb-pool-ip 10.0.0.11`, and `--emb-pool-host 192.168.123.7`.
For a manually launched smoke run, connect from PM2 to the PIM server with
`ssh 192.168.123.7`; `scripts/run_e2e.py` performs this step automatically.

## Quick Artifact Evaluation (Recommended)

The evaluation environment is already configured on PM2 and PM5. After logging
in to PM2, one command checks both servers, builds all four systems, runs one
representative RM1 point per system, and writes a CSV summary under `results/`:

```sh
scripts/run_artifact.sh quick
```

For the complete recommended RM1-RM4 knee matrix, run the optional second
command:

```sh
scripts/run_artifact.sh full
```

The quick run is the recommended sanity check and normally takes a few minutes.
The full run is intended to finish in under about one hour. Both commands manage
the model and embedding-pool processes automatically; evaluators do not need to
start services manually or enter any server addresses, Python paths, or ports.
The final line prints the exact `summary.csv` path.

The remaining sections document environment setup, individual roles, and custom
experiment matrices for advanced use. They are not required for the standard
artifact evaluation.

## Repository Layout

- `cloverrec/`: CloverRec implementation. Important entry files are
  `dlrm_model.py`, `dlrm_emb_pool.py`, `dlrm_coordinator.py`, `pim_dpu.c`,
  `pim_module.cpp`, and `client_cache.cpp`.
- `naive_pim_emb/`: naive PIM embedding baseline with the same model,
  coordinator, embedding-pool, PIM, and client-cache structure.
- `remote_emb/`: remote embedding baseline. It uses a model server,
  coordinator, embedding pool, and client cache, but no PIM DPU module.
- `local_emb/`: local CPU embedding baseline. It uses a model server and
  coordinator only; no embedding-pool process is needed.
- `scripts/`: artifact entry points for environment checks, builds, smoke runs,
  end-to-end sweeps, and result parsing.
- `environment.yml` and `environment-pim.yml`: Conda environments for
  GPU/coordinator machines and the PIM embedding-pool machine.
- `requirements.txt` and `requirements-pim.txt`: pip dependency lists mirrored
  by the Conda environment files.
- `data/Kaggle/`: optional processed Kaggle/Criteo workload files. Synthetic
  `RM1`-`RM4` workloads do not require this directory.
- `results/`: generated logs and summary CSV files from local validation runs.

## Hardware Setup

The end-to-end experiment uses up to three server roles:

- model server: GPU server running `dlrm_model.py`
- coordinator: host process running `dlrm_coordinator.py`
- embedding pool: PIM or remote embedding server running `dlrm_emb_pool.py`

The current lab setup uses two V100 GPU servers and one UPMEM PIM server. Use the
InfiniBand/RDMA IPs for CloverRec commands:

- GPU/model/coordinator candidates: `10.0.0.3` and `10.0.0.5`
- PIM embedding pool: `10.0.0.11`

The evaluator-facing default uses PM2 (`10.0.0.5`) for both the model server and
coordinator, and PM5 (`10.0.0.11`) for the embedding pool. PM1 (`10.0.0.3`) is
an optional GPU candidate and is not required for the default evaluation.

Use the IP address assigned to the InfiniBand/RDMA NIC, not the management NIC.
On most setups this is the address shown on an `ib*`, `ibp*`, or RDMA-backed
interface in `ip -brief addr` / `rdma link`. The coordinator can run on either
GPU server, including the same machine as the model server.

## Advanced: Recreating the Environment

- Ubuntu 22.04 LTS
- Python 3.10
- g++ 11.4 or newer
- NVIDIA driver and a V100-class GPU for the model server
- Mellanox InfiniBand/RDMA stack with `ibverbs` and `pyverbs`
- UPMEM SDK, `dpu-upmem-dpurte-clang`, and `libdpu` for PIM systems

Create the Conda environment on each GPU/coordinator server:

```sh
conda env create -f environment.yml
conda activate CloverRec
```

On the PIM embedding-pool server, use the CPU PyTorch environment to avoid
installing GPU CUDA wheels:

```sh
conda env create -f environment-pim.yml
conda activate CloverRec
```

The scripts set `PYTHONNOUSERSITE=1` by default so that the artifact does not
accidentally use packages from `~/.local`.

`pyverbs` is intentionally treated as an RDMA/system dependency. If it is not
available after creating the Conda environment, install the OS-provided package
or a pyverbs build that matches the server's RDMA stack. On Ubuntu 22.04 without
sudo access, the repository includes a helper that extracts `python3-pyverbs`
into the active Conda environment:

```sh
scripts/install_pyverbs_from_apt.sh
```

If the environment already exists, update it from the repository root:

```sh
conda env update -f environment.yml --prune
conda activate CloverRec
scripts/install_pyverbs_from_apt.sh
```

Use `environment-pim.yml` in the `conda env update` command on the PIM server.

Check each machine:

```sh
scripts/check_env.sh --role model
scripts/check_env.sh --role coordinator
scripts/check_env.sh --role emb_pool
```

## Advanced: Manual Build

Build from the repository root. Build only the system you plan to run on the
current machine.

```sh
scripts/build.sh --system cloverrec
scripts/build.sh --system naive_pim_emb
scripts/build.sh --system remote_emb
scripts/build.sh --system local_emb
```

On a non-PIM machine, `--component auto` skips the PIM module if the UPMEM tools
are not available. To force a component:

```sh
scripts/build.sh --system cloverrec --component client
scripts/build.sh --system cloverrec --component pim
```

## Experiment Runtime Defaults

The default script parameters are chosen to keep artifact evaluation practical
while still showing the end-to-end performance trend. Smoke and matrix runs use
random `RM` workloads by default, run `100` measured batches, and use `100000`
rows per synthetic embedding table. The table size has little effect on the
random-workload performance trend, but `100000` starts much faster than the
original `1000000`. Increase `--num-batches` or `--table-size` for longer
validation experiments. The `KAGGLE` workload uses real processed table counts
and ignores `--table-size`.

Expected runtime on our three-server setup:

- A single smoke coordinator run usually takes a few seconds to a few minutes,
  depending on workload, system, and batch size.
- `scripts/run_e2e.py --batch-profile smoke` runs one quick point per
  `(system, workload)` pair and is the fastest end-to-end sanity check.
- The default `knee` profile over `RM1`-`RM4` and all four systems is intended
  to finish in under about one hour after the environment has already been
  built. In our recent validation, a slightly wider RM4 sweep took about
  48 minutes; the current default knee profile removes several of those longest
  RM4 points. RM4 is the longest workload, and very large RM4 batches are
  deliberately trimmed because they add substantial latency and wall time after
  throughput has mostly saturated.
- Adding `KAGGLE` to the full knee sweep can increase the total runtime to about
  two hours. KAGGLE is useful for checking the real-data path, but each
  coordinator point reloads the 2.3GB processed Kaggle file, so its wall-clock
  time is dominated by data loading rather than the measured inference loop.
- First-time setup can take longer because the Conda environments and native
  extensions must be created and built on the GPU/coordinator and PIM servers.

## Advanced: Manual Role-by-Role Smoke Run

Start the model server on a GPU server:

```sh
scripts/run_smoke.py --system cloverrec --role model --workload RM1
```

Start the embedding pool on the PIM server:

```sh
scripts/run_smoke.py --system cloverrec --role emb_pool --workload RM1
```

Start the coordinator after the two servers are ready:

```sh
scripts/run_smoke.py \
  --system cloverrec \
  --role coordinator \
  --workload RM1 \
  --batch-size 128 \
  --table-size 100000 \
  --model-ip 10.0.0.5 \
  --emb-pool-ip 10.0.0.11
```

The same script supports `RM1`, `RM2`, `RM3`, `RM4`, and `KAGGLE`, and all four
systems:

```sh
scripts/run_smoke.py --system naive_pim_emb --role model --workload RM1
scripts/run_smoke.py --system naive_pim_emb --role emb_pool --workload RM1
scripts/run_smoke.py --system naive_pim_emb --role coordinator --workload RM1

scripts/run_smoke.py --system remote_emb --role model --workload RM1
scripts/run_smoke.py --system remote_emb --role emb_pool --workload RM1
scripts/run_smoke.py --system remote_emb --role coordinator --workload RM1

scripts/run_smoke.py --system local_emb --role model --workload RM1
scripts/run_smoke.py --system local_emb --role coordinator --workload RM1
```

For `local_emb`, no embedding-pool process is needed.

## Advanced: Custom End-to-End Matrix

Use `scripts/run_e2e.py` for the regular end-to-end experiment matrix. It starts
the model server, starts the embedding pool when the selected system needs one,
runs the coordinator, parses the coordinator log, and writes a summary under
`results/e2e/`.

```sh
scripts/run_e2e.py \
  --systems cloverrec \
  --workloads RM1 \
  --batch-sizes 128 \
  --model-ip 10.0.0.5 \
  --emb-pool-ip 10.0.0.11 \
  --emb-pool-host 192.168.123.7 \
  --coordinator-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --model-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --emb-pool-python /home/cml/miniconda3/envs/CloverRec/bin/python
```

`--model-ip` and `--emb-pool-ip` are the InfiniBand/RDMA addresses used by the
runtime. `--model-host` and `--emb-pool-host` are SSH targets used only by the
launcher to start remote processes. Leave `--model-host local` when the model
server runs on the coordinator machine.

When `--batch-sizes` is omitted, `scripts/run_e2e.py` uses the `knee` batch
profile: each `(system, workload)` pair gets a small list of batch sizes chosen
to show the throughput knee without pushing latency far past saturation. Preview
the exact matrix before running:

```sh
scripts/run_e2e.py \
  --systems cloverrec remote_emb naive_pim_emb local_emb \
  --workloads RM1 RM2 RM3 RM4 \
  --print-matrix
```

To sweep the recommended knee matrix:

```sh
scripts/run_e2e.py \
  --systems cloverrec remote_emb naive_pim_emb local_emb \
  --workloads RM1 RM2 RM3 RM4 \
  --num-batches 100 \
  --table-size 100000 \
  --model-ip 10.0.0.5 \
  --emb-pool-ip 10.0.0.11 \
  --emb-pool-host 192.168.123.7 \
  --coordinator-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --model-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --emb-pool-python /home/cml/miniconda3/envs/CloverRec/bin/python
```

For RM2 and RM4 the default profile includes smaller batch sizes than the old
`64 128 256` sweep, because those workloads tend to reach saturation earlier and
large batches mostly add latency. Pass `--batch-sizes` to force one global list,
or use `--batch-profile smoke` for one quick point per pair and
`--batch-profile wide` for a broader exploratory sweep.

To include the optional processed Kaggle/Criteo workload, add `KAGGLE` to
`--workloads`. This is mainly a real-data-path check and is slower than the
default synthetic RM sweep because every coordinator process reloads the
processed dataset. The default path is `data/Kaggle`; override it when needed:

```sh
scripts/run_e2e.py \
  --systems cloverrec remote_emb naive_pim_emb local_emb \
  --workloads KAGGLE \
  --kaggle-data-root data/Kaggle \
  --model-ip 10.0.0.5 \
  --emb-pool-ip 10.0.0.11 \
  --emb-pool-host 192.168.123.7 \
  --coordinator-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --model-python /home/cml/anaconda3/envs/CloverRec/bin/python \
  --emb-pool-python /home/cml/miniconda3/envs/CloverRec/bin/python
```

For PIM systems, run this after syncing the repository to the PIM server and
creating the `environment-pim.yml` environment there. The script can build
components automatically; pass `--skip-build` to reuse existing native modules.

## Advanced: CloverRec PIM Parameters

The adaptive CloverRec PIM parameters can be passed through the embedding-pool
role. Defaults match the previous hard-coded implementation.

```sh
scripts/run_smoke.py \
  --system cloverrec \
  --role emb_pool \
  --workload RM1 \
  --redundant-ratio 0.005 \
  --split-emb-num-ratio 0.0005 \
  --merge-emb-num-ratio 0.0001 \
  --select-emb-iter 300 \
  --aging-freq 100 \
  --delta-freq 100
```

These options map to the hot-embedding replication budget, split/merge rate, hot
embedding sampling window, frequency aging interval, and adjustment interval.

## Advanced: Parsing Results

Save coordinator output and parse it into JSON or CSV:

```sh
scripts/run_smoke.py --system cloverrec --role coordinator --workload RM1 \
  --model-ip 10.0.0.5 --emb-pool-ip 10.0.0.11 | tee cloverrec_rm1.log

scripts/parse_results.py cloverrec_rm1.log
scripts/parse_results.py --format csv cloverrec_rm1.log
```

The parser extracts throughput, average latency, embedding lookup time, apply
embedding time, and available breakdown fields from the standard output.

## Dataset Notes

The `KAGGLE` workload defaults to `data/Kaggle`, with
`train.txt` as the raw-data path stem and
`kaggleAdDisplayChallenge_processed.npz` as the processed file. Override these
with `--kaggle-data-root`, `--kaggle-raw-data-file`, or
`--kaggle-processed-data-file`. The raw `train.txt` file is only needed if the
dataset must be preprocessed again; for normal runs the processed `.npz` file
and `train_day_count.npz` / `train_fea_count.npz` are sufficient.
