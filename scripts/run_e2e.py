#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from parse_results import parse_file
from run_smoke import DEFAULT_KAGGLE_DATA_ROOT, WORKLOADS, workload_config


SYSTEMS = ["cloverrec", "local_emb", "remote_emb", "naive_pim_emb"]
KNEE_BATCH_SIZES = {
    ("cloverrec", "RM1"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("remote_emb", "RM1"): [1, 2, 4, 8, 16, 32, 64],
    ("naive_pim_emb", "RM1"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("local_emb", "RM1"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("cloverrec", "RM2"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("remote_emb", "RM2"): [1, 2, 4, 8, 16, 32],
    ("naive_pim_emb", "RM2"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("local_emb", "RM2"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("cloverrec", "RM3"): [1, 2, 4, 8, 16, 32, 64, 128, 256],
    ("remote_emb", "RM3"): [1, 2, 4, 8, 16, 32, 64, 128],
    ("naive_pim_emb", "RM3"): [1, 2, 4, 8, 16, 32, 64, 128, 256],
    ("local_emb", "RM3"): [1, 2, 4, 8, 16, 32, 64, 128, 256],
    ("cloverrec", "RM4"): [1, 2, 4, 8, 16, 32],
    ("remote_emb", "RM4"): [1, 2, 4, 8, 16],
    ("naive_pim_emb", "RM4"): [1, 2, 4, 8, 16, 32],
    ("local_emb", "RM4"): [1, 2, 4, 8, 16, 32],
    ("cloverrec", "KAGGLE"): [1, 2, 4, 8, 16, 32, 64],
    ("remote_emb", "KAGGLE"): [1, 2, 4, 8, 16, 32, 64],
    ("naive_pim_emb", "KAGGLE"): [1, 2, 4, 8, 16, 32, 64],
    ("local_emb", "KAGGLE"): [1, 2, 4, 8, 16, 32, 64],
}
SMOKE_BATCH_SIZES = {
    ("*", "KAGGLE"): [4],
    ("*", "*"): [64],
}
WIDE_BATCH_SIZES = {
    ("*", "RM4"): [4, 8, 16, 32, 64, 128],
    ("*", "KAGGLE"): [4, 8, 16, 32, 64],
    ("*", "*"): [16, 32, 64, 128, 256, 512],
}
BATCH_SIZE_PROFILES = {
    "knee": KNEE_BATCH_SIZES,
    "smoke": SMOKE_BATCH_SIZES,
    "wide": WIDE_BATCH_SIZES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CloverRec end-to-end experiment matrix")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["cloverrec"])
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=["RM1"])
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Override the batch profile with one global batch-size list.",
    )
    parser.add_argument(
        "--batch-profile",
        choices=sorted(BATCH_SIZE_PROFILES),
        default=os.environ.get("BATCH_PROFILE", "knee"),
        help="Per-(system, workload) batch-size profile used when --batch-sizes is omitted.",
    )
    parser.add_argument("--num-batches", type=int, default=int(os.environ.get("NUM_BATCHES", "100")))
    parser.add_argument("--table-size", type=int, default=int(os.environ.get("TABLE_SIZE", "100000")))
    parser.add_argument("--zipf", default="1.5")
    parser.add_argument("--model-ip", default=os.environ.get("MODEL_SERVER_IP", "10.0.0.5"))
    parser.add_argument("--emb-pool-ip", default=os.environ.get("EMB_POOL_IP", "10.0.0.11"))
    parser.add_argument("--model-host", default=os.environ.get("MODEL_HOST", "local"))
    parser.add_argument("--emb-pool-host", default=os.environ.get("EMB_POOL_HOST", "192.168.123.7"))
    parser.add_argument("--remote-repo-root", default=os.environ.get("REMOTE_REPO_ROOT", "/home/cml/CloverRec"))
    parser.add_argument("--coordinator-python", default=os.environ.get("COORDINATOR_PYTHON", sys.executable))
    parser.add_argument("--model-python", default=os.environ.get("MODEL_PYTHON", sys.executable))
    parser.add_argument("--emb-pool-python", default=os.environ.get("EMB_POOL_PYTHON", sys.executable))
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--rdma-wr-capacity", default=os.environ.get("RDMA_WR_CAPACITY", "128"))
    parser.add_argument("--rdma-mr-size-mb", default=os.environ.get("RDMA_MR_SIZE_MB", "128"))
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
    parser.add_argument("--base-model-port", type=int, default=18000)
    parser.add_argument("--base-emb-pool-port", type=int, default=19000)
    parser.add_argument("--startup-wait", type=float, default=5.0)
    parser.add_argument("--logs-dir", type=Path, default=Path("results/e2e"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-servers", action="store_true")
    parser.add_argument(
        "--restart-per-case",
        action="store_true",
        help="Restart model and embedding-pool servers for every matrix point.",
    )
    parser.add_argument(
        "--model-system",
        choices=SYSTEMS,
        default=os.environ.get("MODEL_SYSTEM", "remote_emb"),
        help="System directory used for the shared model server.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print the planned matrix and exit without building or running.",
    )
    parser.add_argument("--format", choices=["json", "csv"], default="csv")

    parser.add_argument("--redundant-ratio", default="0.005")
    parser.add_argument("--split-emb-num-ratio", default="0.0005")
    parser.add_argument("--merge-emb-num-ratio", default="0.0001")
    parser.add_argument("--select-emb-iter", default="300")
    parser.add_argument("--aging-freq", default="100")
    parser.add_argument("--delta-freq", default="100")

    return parser.parse_args()


def batch_sizes_for(args: argparse.Namespace, system: str, workload: str) -> list[int]:
    if args.batch_sizes:
        return args.batch_sizes

    profile = BATCH_SIZE_PROFILES[args.batch_profile]
    for key in ((system, workload), ("*", workload), (system, "*"), ("*", "*")):
        if key in profile:
            return profile[key]
    raise KeyError(f"No batch sizes configured for {system} {workload}")


def first_batch_size_for_workload(args: argparse.Namespace, workload: str) -> int:
    return batch_sizes_for(args, args.systems[0], workload)[0]


def print_matrix(args: argparse.Namespace) -> None:
    source = "manual --batch-sizes" if args.batch_sizes else f"{args.batch_profile} profile"
    total = 0
    print(f"Batch-size source: {source}")
    for workload in args.workloads:
        print(f"\n{workload}")
        for system in args.systems:
            sizes = batch_sizes_for(args, system, workload)
            total += len(sizes)
            print(f"  {system}: {' '.join(str(size) for size in sizes)}")
    print(f"\nTotal coordinator runs: {total}")


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check)


def remote_cmd(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command]


def is_remote(host: str) -> bool:
    return host not in ("", "local", "localhost", "127.0.0.1")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def start_process(cmd: list[str], log_path: Path, cwd: Path | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    print("+ " + " ".join(shlex.quote(part) for part in cmd) + f" > {log_path} 2>&1", flush=True)
    return subprocess.Popen(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def terminate_process(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    print(f"Stopping {label}", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def cleanup_remote_process(host: str, system: str, role: str, port: int) -> None:
    if not is_remote(host):
        return

    script = "dlrm_model.py" if role == "model" else "dlrm_emb_pool.py"
    port_arg = "--server-port" if role == "model" else "--emb-pool-port"
    patterns = [
        f"{script} .*{port_arg} {port}",
        f"run_smoke.py --system {system} --role {role}.*{port_arg} {port}",
    ]
    command = "; ".join(f"pkill -f {shlex.quote(pattern)} 2>/dev/null || true" for pattern in patterns)
    subprocess.run(remote_cmd(host, command), check=False)


def build_system(args: argparse.Namespace, repo_root: Path, system: str) -> None:
    if args.skip_build:
        return

    local_client_build = f"cd {shlex.quote(str(repo_root))} && PYTHON={shlex.quote(args.coordinator_python)} scripts/build.sh --system {system} --component client"
    run_cmd(["bash", "-lc", local_client_build])

    model_repo = args.remote_repo_root if is_remote(args.model_host) else str(repo_root)
    client_build = f"cd {shlex.quote(model_repo)} && PYTHON={shlex.quote(args.model_python)} scripts/build.sh --system {system} --component client"
    if is_remote(args.model_host):
        run_cmd(remote_cmd(args.model_host, client_build))
    elif Path(args.model_python).resolve() != Path(args.coordinator_python).resolve():
        run_cmd(["bash", "-lc", client_build])

    if system in ("cloverrec", "naive_pim_emb"):
        emb_repo = args.remote_repo_root if is_remote(args.emb_pool_host) else str(repo_root)
        command = f"cd {shlex.quote(emb_repo)} && PYTHON={shlex.quote(args.emb_pool_python)} scripts/build.sh --system {system} --component pim"
        if is_remote(args.emb_pool_host):
            run_cmd(remote_cmd(args.emb_pool_host, command))
        else:
            run_cmd(["bash", "-lc", command])


def make_role_cmd(
    args: argparse.Namespace,
    repo_root: Path,
    system: str,
    role: str,
    workload: str,
    batch_size: int,
    model_port: int,
    emb_pool_port: int,
    python_bin: str,
) -> list[str]:
    cmd = [
        python_bin,
        str(repo_root / "scripts" / "run_smoke.py"),
        "--system",
        system,
        "--role",
        role,
        "--workload",
        workload,
        "--batch-size",
        str(batch_size),
        "--num-batches",
        str(args.num_batches),
        "--table-size",
        str(args.table_size),
        "--zipf",
        args.zipf,
        "--model-ip",
        args.model_ip,
        "--emb-pool-ip",
        args.emb_pool_ip,
        "--model-port",
        str(model_port),
        "--emb-pool-port",
        str(emb_pool_port),
        "--gpu",
        args.gpu,
        "--rdma-wr-capacity",
        str(args.rdma_wr_capacity),
        "--rdma-mr-size-mb",
        str(args.rdma_mr_size_mb),
        "--python",
        python_bin,
    ]

    if workload == "KAGGLE":
        cmd.extend(["--kaggle-data-root", args.kaggle_data_root])
        if args.kaggle_raw_data_file:
            cmd.extend(["--kaggle-raw-data-file", args.kaggle_raw_data_file])
        if args.kaggle_processed_data_file:
            cmd.extend(["--kaggle-processed-data-file", args.kaggle_processed_data_file])
        if args.kaggle_no_multiprocessing:
            cmd.append("--kaggle-no-multiprocessing")

    if system == "cloverrec":
        cmd.extend(
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

    return cmd


def start_remote_process(host: str, command: str, log_path: Path) -> subprocess.Popen:
    remote = (
        f"mkdir -p {shlex.quote(str(log_path.parent))} && "
        f"{command} > {shlex.quote(str(log_path))} 2>&1"
    )
    return start_process(remote_cmd(host, remote), Path(os.devnull))


def start_role_process(
    args: argparse.Namespace,
    repo_root: Path,
    system: str,
    role: str,
    workload: str,
    batch_size: int,
    model_port: int,
    emb_pool_port: int,
    log_path: Path,
) -> subprocess.Popen:
    host = args.model_host if role == "model" else args.emb_pool_host
    python_bin = args.model_python if role == "model" else args.emb_pool_python
    role_repo = Path(args.remote_repo_root) if is_remote(host) else repo_root
    role_cmd = make_role_cmd(
        args,
        role_repo,
        system,
        role,
        workload,
        batch_size,
        model_port,
        emb_pool_port,
        python_bin,
    )
    if is_remote(host):
        command = f"cd {shlex.quote(args.remote_repo_root)} && {shell_join(role_cmd)}"
        return start_remote_process(host, command, log_path)
    return start_process(role_cmd, log_path, cwd=repo_root)


def assert_running(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        raise RuntimeError(f"{label} exited during startup")


def case_tag(
    system: str,
    workload: str,
    batch_size: int,
    num_batches: int,
    table_size: int,
) -> str:
    return f"{system}_{workload}_bs{batch_size}_nb{num_batches}_ts{table_size}"


def error_row(
    args: argparse.Namespace,
    system: str,
    workload: str,
    batch_size: int,
    model_port: int,
    emb_pool_port: int | None,
    exc: Exception,
    coord_log: Path | None = None,
) -> dict:
    row = {
        "system": system,
        "workload": workload,
        "batch_size": batch_size,
        "num_batches": args.num_batches,
        "table_size": args.table_size,
        "model_port": model_port,
        "emb_pool_port": emb_pool_port,
        "error": str(exc),
    }
    if coord_log is not None:
        row["file"] = str(coord_log)
    return row


def run_coordinator(
    args: argparse.Namespace,
    repo_root: Path,
    system: str,
    workload: str,
    batch_size: int,
    model_port: int,
    emb_pool_port: int,
) -> dict:
    tag = case_tag(system, workload, batch_size, args.num_batches, args.table_size)
    run_dir = args.logs_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    coord_log = run_dir / "coordinator.log"

    coord_cmd = make_role_cmd(
        args,
        repo_root,
        system,
        "coordinator",
        workload,
        batch_size,
        model_port,
        emb_pool_port,
        args.coordinator_python,
    )
    with coord_log.open("w") as log_file:
        print("+ " + shell_join(coord_cmd) + f" > {coord_log} 2>&1", flush=True)
        start_time = time.monotonic()
        coord = subprocess.run(
            coord_cmd,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        coordinator_wall_time_sec = time.monotonic() - start_time
    if coord.returncode != 0:
        raise RuntimeError(f"coordinator failed with exit code {coord.returncode}")

    row = parse_file(coord_log)
    row.update(
        {
            "system": system,
            "workload": workload,
            "batch_size": batch_size,
            "num_batches": args.num_batches,
            "table_size": args.table_size,
            "model_port": model_port,
            "emb_pool_port": None if system == "local_emb" else emb_pool_port,
            "coordinator_wall_time_sec": round(coordinator_wall_time_sec, 3),
        }
    )
    return row


def run_one(
    args: argparse.Namespace,
    repo_root: Path,
    system: str,
    workload: str,
    batch_size: int,
    index: int,
) -> dict:
    cfg = workload_config(workload, args.table_size)
    model_port = args.base_model_port + index
    emb_pool_port = args.base_emb_pool_port + index
    tag = case_tag(system, workload, batch_size, args.num_batches, args.table_size)
    run_dir = args.logs_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {tag} ===", flush=True)

    model_log = run_dir / "model.log"
    emb_log = run_dir / "emb_pool.log"
    coord_log = run_dir / "coordinator.log"
    servers: list[tuple[subprocess.Popen, str]] = []

    try:
        servers.append(
            (
                start_role_process(
                    args,
                    repo_root,
                    system,
                    "model",
                    workload,
                    batch_size,
                    model_port,
                    emb_pool_port,
                    model_log,
                ),
                "model",
            )
        )

        if system != "local_emb":
            servers.append(
                (
                    start_role_process(
                        args,
                        repo_root,
                        system,
                        "emb_pool",
                        workload,
                        batch_size,
                        model_port,
                        emb_pool_port,
                        emb_log,
                    ),
                    "emb_pool",
                )
            )

        time.sleep(args.startup_wait)

        for proc, label in servers:
            assert_running(proc, label)

        return run_coordinator(args, repo_root, system, workload, batch_size, model_port, emb_pool_port)
    finally:
        if not args.keep_servers:
            for proc, label in reversed(servers):
                terminate_process(proc, label)
            cleanup_remote_process(args.model_host, system, "model", model_port)
            if system != "local_emb":
                cleanup_remote_process(args.emb_pool_host, system, "emb_pool", emb_pool_port)


def run_reused_servers(args: argparse.Namespace, repo_root: Path, summary_path: Path) -> list[dict]:
    rows: list[dict] = []
    model_index = 0
    emb_pool_index = 0

    for workload in args.workloads:
        cfg = workload_config(workload, args.table_size)
        del cfg

        model_batch_size = first_batch_size_for_workload(args, workload)
        model_port = args.base_model_port + model_index
        model_index += 1
        model_dir = args.logs_dir / "_servers" / f"{workload}_model"
        model_log = model_dir / "model.log"
        print(f"\n=== shared_model_{workload} ===", flush=True)
        model_proc = start_role_process(
            args,
            repo_root,
            args.model_system,
            "model",
            workload,
            model_batch_size,
            model_port,
            args.base_emb_pool_port,
            model_log,
        )
        try:
            time.sleep(args.startup_wait)
            assert_running(model_proc, "model")

            for system in args.systems:
                emb_pool_port = args.base_emb_pool_port + emb_pool_index
                emb_proc: subprocess.Popen | None = None
                batch_sizes = batch_sizes_for(args, system, workload)

                try:
                    if system != "local_emb":
                        emb_pool_index += 1
                        emb_dir = args.logs_dir / "_servers" / f"{system}_{workload}_emb_pool"
                        emb_log = emb_dir / "emb_pool.log"
                        print(f"\n=== shared_emb_pool_{system}_{workload} ===", flush=True)
                        emb_proc = start_role_process(
                            args,
                            repo_root,
                            system,
                            "emb_pool",
                            workload,
                            batch_sizes[0],
                            model_port,
                            emb_pool_port,
                            emb_log,
                        )
                        time.sleep(args.startup_wait)
                        assert_running(emb_proc, "emb_pool")

                    for batch_size in batch_sizes:
                        tag = case_tag(
                            system,
                            workload,
                            batch_size,
                            args.num_batches,
                            args.table_size,
                        )
                        coord_log = args.logs_dir / tag / "coordinator.log"
                        print(f"\n=== {tag} ===", flush=True)
                        try:
                            assert_running(model_proc, "model")
                            if emb_proc is not None:
                                assert_running(emb_proc, "emb_pool")
                            rows.append(
                                run_coordinator(
                                    args,
                                    repo_root,
                                    system,
                                    workload,
                                    batch_size,
                                    model_port,
                                    emb_pool_port,
                                )
                            )
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            rows.append(
                                error_row(
                                    args,
                                    system,
                                    workload,
                                    batch_size,
                                    model_port,
                                    None if system == "local_emb" else emb_pool_port,
                                    exc,
                                    coord_log,
                                )
                            )
                            print(
                                f"ERROR: {system} {workload} batch_size={batch_size}: {exc}",
                                flush=True,
                            )
                        write_summary(rows, summary_path, args.format, echo=False)
                finally:
                    if not args.keep_servers and emb_proc is not None:
                        terminate_process(emb_proc, "emb_pool")
                        cleanup_remote_process(args.emb_pool_host, system, "emb_pool", emb_pool_port)
        finally:
            if not args.keep_servers:
                terminate_process(model_proc, "model")
                cleanup_remote_process(args.model_host, args.model_system, "model", model_port)

    return rows


def run_restart_per_case(args: argparse.Namespace, repo_root: Path, summary_path: Path) -> list[dict]:
    rows = []
    run_index = 0
    for system in args.systems:
        for workload in args.workloads:
            for batch_size in batch_sizes_for(args, system, workload):
                model_port = args.base_model_port + run_index
                emb_pool_port = None if system == "local_emb" else args.base_emb_pool_port + run_index
                coord_log = (
                    args.logs_dir
                    / case_tag(system, workload, batch_size, args.num_batches, args.table_size)
                    / "coordinator.log"
                )
                try:
                    rows.append(run_one(args, repo_root, system, workload, batch_size, run_index))
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append(
                        error_row(
                            args,
                            system,
                            workload,
                            batch_size,
                            model_port,
                            emb_pool_port,
                            exc,
                            coord_log,
                        )
                    )
                    print(f"ERROR: {system} {workload} batch_size={batch_size}: {exc}", flush=True)
                write_summary(rows, summary_path, args.format, echo=False)
                run_index += 1

    return rows


def write_summary(rows: list[dict], output_path: Path, fmt: str, echo: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        if echo:
            print(json.dumps(rows, indent=2, sort_keys=True))
        return

    keys = [
        "system",
        "workload",
        "batch_size",
        "num_batches",
        "table_size",
        "throughput_req_per_sec",
        "avg_latency_ms",
        "avg_ev_lookup_ms",
        "avg_apply_emb_ms",
        "cpu_cal_ms",
        "pim_ev_trans_ms",
        "ev_transmission_ms",
        "network_gpu_ms",
        "others_ms",
        "time_elapsed_sec",
        "coordinator_wall_time_sec",
        "num_latency_samples",
        "model_port",
        "emb_pool_port",
        "error",
        "file",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if echo:
        with output_path.open() as stream:
            print(stream.read(), end="")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.logs_dir = args.logs_dir.resolve()

    if args.print_matrix:
        print_matrix(args)
        return 0

    suffix = "json" if args.format == "json" else "csv"
    summary_path = args.logs_dir / f"summary.{suffix}"
    for system in args.systems:
        build_system(args, repo_root, system)

    if args.model_system not in args.systems:
        build_system(args, repo_root, args.model_system)

    if args.restart_per_case:
        rows = run_restart_per_case(args, repo_root, summary_path)
    else:
        rows = run_reused_servers(args, repo_root, summary_path)

    write_summary(rows, summary_path, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
