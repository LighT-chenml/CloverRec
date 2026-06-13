#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


SYSTEMS = ["cloverrec", "local_emb", "remote_emb", "naive_pim_emb"]
WORKLOADS = ["RM1", "RM2", "RM3", "RM4", "KAGGLE"]
DEFAULT_KAGGLE_DATA_ROOT = "data/Kaggle"
KAGGLE_EMB = (
    "1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-"
    "27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"
)


def repeat_value(value: int, count: int) -> str:
    return "-".join([str(value)] * count)


def workload_config(workload: str, table_size: int) -> dict[str, str]:
    emb_10 = repeat_value(table_size, 10)
    emb_40 = repeat_value(table_size, 40)
    emb_80 = repeat_value(table_size, 80)

    configs = {
        "RM1": {
            "model_port": "8000",
            "emb_pool_port": "1234",
            "bot_mlp": "256-128-64",
            "top_mlp": "256-64-1",
            "num_int": "119",
            "emb_size": "64",
            "num_indices": "80",
            "emb": emb_10,
            "model_emb": emb_10,
        },
        "RM2": {
            "model_port": "8001",
            "emb_pool_port": "1235",
            "bot_mlp": "256-128-64",
            "top_mlp": "512-128-1",
            "num_int": "884",
            "emb_size": "64",
            "num_indices": "80",
            "emb": emb_40,
            "model_emb": emb_10,
        },
        "RM3": {
            "model_port": "8002",
            "emb_pool_port": "1236",
            "bot_mlp": "2560-512-64",
            "top_mlp": "512-128-1",
            "num_int": "119",
            "emb_size": "64",
            "num_indices": "20",
            "emb": emb_10,
            "model_emb": emb_10,
        },
        "RM4": {
            "model_port": "8003",
            "emb_pool_port": "1237",
            "bot_mlp": "512-256-128",
            "top_mlp": "512-128-1",
            "num_int": "3368",
            "emb_size": "128",
            "num_indices": "160",
            "emb": emb_80,
            "model_emb": emb_10,
        },
        "KAGGLE": {
            "model_port": "8004",
            "emb_pool_port": "1238",
            "data_generation": "dataset",
            "data_set": "kaggle",
            "model_data_generation": "dataset",
            "bot_mlp": "13-512-256-64",
            "top_mlp": "512-256-1",
            "num_int": "415",
            "emb_size": "64",
            "num_indices": "80",
            "emb": KAGGLE_EMB,
            "model_emb": repeat_value(1000000, 10),
        },
    }
    return configs[workload]


def kaggle_paths(args: argparse.Namespace) -> tuple[str, str]:
    data_root = Path(args.kaggle_data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[1] / data_root
    raw_data_file = args.kaggle_raw_data_file or str(data_root / "train.txt")
    processed_data_file = args.kaggle_processed_data_file or str(
        data_root / "kaggleAdDisplayChallenge_processed.npz"
    )
    return raw_data_file, processed_data_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch CloverRec smoke-test roles on one machine."
    )
    parser.add_argument("--system", choices=SYSTEMS, default="cloverrec")
    parser.add_argument("--role", choices=["model", "emb_pool", "coordinator"], required=True)
    parser.add_argument("--workload", choices=WORKLOADS, default="RM1")
    parser.add_argument("--batch-size", default="128")
    parser.add_argument("--num-batches", default=os.environ.get("NUM_BATCHES", "100"))
    parser.add_argument(
        "--table-size",
        type=int,
        default=int(os.environ.get("TABLE_SIZE", "100000")),
        help="Rows per synthetic embedding table. 100000 starts much faster than 1000000.",
    )
    parser.add_argument("--zipf", default="1.5")
    parser.add_argument("--model-ip", default=os.environ.get("MODEL_SERVER_IP", "10.0.0.5"))
    parser.add_argument("--emb-pool-ip", default=os.environ.get("EMB_POOL_IP", "10.0.0.11"))
    parser.add_argument("--model-port")
    parser.add_argument("--emb-pool-port")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--rdma-wr-capacity", default=os.environ.get("RDMA_WR_CAPACITY", "128"))
    parser.add_argument("--rdma-mr-size-mb", default=os.environ.get("RDMA_MR_SIZE_MB", "128"))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument(
        "--kaggle-data-root",
        default=os.environ.get("KAGGLE_DATA_ROOT", DEFAULT_KAGGLE_DATA_ROOT),
        help="Directory containing Kaggle processed files.",
    )
    parser.add_argument(
        "--kaggle-raw-data-file",
        default=os.environ.get("KAGGLE_RAW_DATA_FILE"),
        help="Kaggle raw path used to derive train_day_count/train_fea_count names.",
    )
    parser.add_argument(
        "--kaggle-processed-data-file",
        default=os.environ.get("KAGGLE_PROCESSED_DATA_FILE"),
        help="Path to kaggleAdDisplayChallenge_processed.npz.",
    )
    parser.add_argument(
        "--kaggle-no-multiprocessing",
        action="store_true",
        help="Do not pass --dataset-multiprocessing for the Kaggle workload.",
    )

    parser.add_argument("--redundant-ratio", default="0.005")
    parser.add_argument("--split-emb-num-ratio", default="0.0005")
    parser.add_argument("--merge-emb-num-ratio", default="0.0001")
    parser.add_argument("--select-emb-iter", default="300")
    parser.add_argument("--aging-freq", default="100")
    parser.add_argument("--delta-freq", default="100")

    return parser.parse_args()


def base_model_args(args: argparse.Namespace, cfg: dict[str, str], model_port: str) -> list[str]:
    return [
        "--num-batches",
        args.num_batches,
        "--data-generation",
        cfg.get("model_data_generation", cfg.get("data_generation", "random")),
        "--num-int",
        cfg["num_int"],
        "--arch-mlp-bot",
        cfg["bot_mlp"],
        "--arch-mlp-top",
        cfg["top_mlp"],
        "--arch-sparse-feature-size",
        cfg["emb_size"],
        "--arch-embedding-size",
        cfg["model_emb"],
        "--num-indices-per-lookup",
        cfg["num_indices"],
        "--num-indices-per-lookup-fixed=True",
        "--arch-interaction-op",
        "dot",
        "--numpy-rand-seed",
        "727",
        "--print-freq",
        "10",
        "--print-time",
        "--inference-only",
        "--get-cdf-lat=True",
        "--use-gpu",
        "--server-port",
        model_port,
        "--rdma-mr-size-mb",
        args.rdma_mr_size_mb,
    ]


def base_emb_pool_args(args: argparse.Namespace, cfg: dict[str, str], emb_pool_port: str) -> list[str]:
    cmd_args = [
        "--arch-sparse-feature-size",
        cfg["emb_size"],
        "--arch-embedding-size",
        cfg["emb"],
        "--rdma-wr-capacity",
        args.rdma_wr_capacity,
        "--emb-pool-port",
        emb_pool_port,
    ]

    if args.system in ("cloverrec", "naive_pim_emb"):
        cmd_args.extend(["--rdma-mr-size-mb", args.rdma_mr_size_mb])

    if args.system == "cloverrec":
        cmd_args.extend(
            [
                "--redundant-ratio",
                args.redundant_ratio,
                "--split-emb-num-ratio",
                args.split_emb_num_ratio,
                "--merge-emb-num-ratio",
                args.merge_emb_num_ratio,
                "--select-emb-iter",
                args.select_emb_iter,
                "--aging-freq",
                args.aging_freq,
                "--delta-freq",
                args.delta_freq,
            ]
        )

    return cmd_args


def base_coordinator_args(
    args: argparse.Namespace, cfg: dict[str, str], model_port: str, emb_pool_port: str
) -> list[str]:
    cmd_args = [
        "--num-batches",
        args.num_batches,
        "--data-generation",
        cfg.get("data_generation", "random"),
        "--rand-data-dist=zipfian",
        "--arch-mlp-bot",
        cfg["bot_mlp"],
        "--arch-mlp-top",
        cfg["top_mlp"],
        "--arch-sparse-feature-size",
        cfg["emb_size"],
        "--arch-embedding-size",
        cfg["emb"],
        "--num-indices-per-lookup",
        cfg["num_indices"],
        "--num-indices-per-lookup-fixed=True",
        "--arch-interaction-op",
        "dot",
        "--numpy-rand-seed",
        "727",
        "--print-freq",
        "10",
        "--print-time",
        "--inference-only",
        "--get-cdf-lat=True",
        "--server-ip",
        args.model_ip,
        "--server-port",
        model_port,
        "--rdma-mr-size-mb",
        args.rdma_mr_size_mb,
        "--zipf-parameter",
        args.zipf,
        "--mini-batch-size",
        args.batch_size,
    ]

    if cfg.get("data_generation") == "dataset":
        raw_data_file, processed_data_file = kaggle_paths(args)
        cmd_args.extend(
            [
                "--data-set",
                cfg["data_set"],
                "--raw-data-file",
                raw_data_file,
                "--processed-data-file",
                processed_data_file,
            ]
        )
        if not args.kaggle_no_multiprocessing:
            cmd_args.append("--dataset-multiprocessing")

    if args.system != "local_emb":
        cmd_args.extend(
            [
                "--emb-pool-ip",
                args.emb_pool_ip,
                "--emb-pool-port",
                emb_pool_port,
                "--rdma-wr-capacity",
                args.rdma_wr_capacity,
            ]
        )

    return cmd_args


def run() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    system_dir = repo_root / args.system
    cfg = workload_config(args.workload, args.table_size)
    model_port = args.model_port or cfg["model_port"]
    emb_pool_port = args.emb_pool_port or cfg["emb_pool_port"]

    if args.role == "emb_pool" and args.system == "local_emb":
        print("local_emb does not use a separate emb_pool role", file=sys.stderr)
        return 2

    print(f"System: {args.system}")
    print(f"Role: {args.role}")
    print(f"Workload: {args.workload}")
    print(f"Num batches: {args.num_batches}")
    if cfg.get("data_generation") == "dataset":
        if args.role == "coordinator":
            raw_data_file, processed_data_file = kaggle_paths(args)
            print(f"Kaggle raw data file: {raw_data_file}")
            print(f"Kaggle processed data file: {processed_data_file}")
        else:
            print("Kaggle workload: dataset paths are used by the coordinator role")
    else:
        print(f"Table size: {args.table_size}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONNOUSERSITE", "1")

    if args.role == "model":
        target = "dlrm_model.py"
        cmd_args = base_model_args(args, cfg, model_port)
        print(f"Starting model server on port {model_port}")
    elif args.role == "emb_pool":
        target = "dlrm_emb_pool.py"
        cmd_args = base_emb_pool_args(args, cfg, emb_pool_port)
        print(f"Starting embedding pool on port {emb_pool_port}")
    else:
        target = "dlrm_coordinator.py"
        cmd_args = base_coordinator_args(args, cfg, model_port, emb_pool_port)
        print(f"Connecting to model server {args.model_ip}:{model_port}")
        if args.system != "local_emb":
            print(f"Connecting to embedding pool {args.emb_pool_ip}:{emb_pool_port}")

    cmd = [args.python, target, *cmd_args]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=system_dir, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(run())
