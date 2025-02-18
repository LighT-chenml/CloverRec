import subprocess
import argparse
import asyncio

import os
import signal

def start_coordinator(model, batch_size):
    print(f"begin test {model} {batch_size}")
    
    command = f"./benchmark_coordinator_{model}.sh {batch_size}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

def start_emb_pool(model):
    command = f"./benchmark_emb_pool_{model}.sh"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print(f"Started emb pool with PID: {process.pid}")
    
    return process.pid

def close_emb_pool():
    command = f"pkill -f dlrm_emb_pool.py"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Emb pool has been terminated.")

def clean_env():
    command = f"pkill -f dlrm"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"All clean.")

def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="CloverRec Benchmarking"
    )
    # model related parameters
    parser.add_argument("--arch-batch-size", type=int, default=2)
    parser.add_argument("--model", type=str, default="RM1")
    
    global args
    args = parser.parse_args()

    models = [args.model]
    # models = ['RM2', 'RM3', 'RM4', 'kaggle']
    batch_sizes = {}
    batch_sizes['RM1'] = [1, 2, 4, 8, 16, 32 ,64]
    batch_sizes['RM2'] = [1, 2, 4, 8, 16, 32, 64]
    batch_sizes['RM3'] = [1, 2, 4, 8, 16, 32, 64]
    batch_sizes['RM4'] = [1, 2, 4, 8, 16, 32, 64]
    batch_sizes['kaggle'] = [1, 2, 4, 8, 16, 32, 64]
    
    for model in models:
        
        print("----------------------------------------------------------------")
        print("current model: " + model)
        print("----------------------------------------------------------------")
        
        # emb_pool_process_id = start_emb_pool(model)
        
        for batch_size in batch_sizes[model]:
            start_coordinator(model, batch_size)
        
        clean_env()            

if __name__ == "__main__":
    run()