import subprocess
import argparse
import asyncio

def start_GPU_server(batch_size):
    commands = 'ssh cml@192.168.123.4 "source ~/.bashrc; cd /home/cml/CloverRec/local_emb; chmod +x ./benchmark_model.sh ; ./benchmark_model.sh 32" '
    # subprocess.run(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(commands, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

def close_GPU_server():
    proc_name="CloverRec_GPU_server"
    command = f"pkill {proc_name}"

    GPU_server = "cml@192.168.123.4" # PM2
    ssh_command = f"ssh {GPU_server} {command}"
    result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

def start_coordinator(batch_size):
    command = f"./benchmark_coordinator.sh {batch_size}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="CloverRec Benchmarking"
    )
    # model related parameters
    parser.add_argument("--arch-batch-size", type=int, default=2)

    batch_sizes = [32]
    for batch_size in batch_sizes:
        start_GPU_server(batch_size)
        # start_coordinator(batch_size)
        # close_GPU_server()

if __name__ == "__main__":
    run()