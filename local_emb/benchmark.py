import subprocess
import argparse
import asyncio

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

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256] # RM 1
    
    for batch_size in batch_sizes:
        start_coordinator(batch_size)

if __name__ == "__main__":
    run()