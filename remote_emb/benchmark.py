import subprocess
import argparse
import asyncio

import os
import signal

def start_coordinator(model, batch_size, zipf_parameter):
    print(f"begin test {model} {batch_size} {zipf_parameter}")
    
    command = f"./benchmark_coordinator_{model}.sh {batch_size} {zipf_parameter}"
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

    # models = [args.model]
    models = ['RM1', 'RM2', 'RM3', 'RM4', 'kaggle']
    batch_sizes = {}
    batch_sizes['RM1'] = [1, 2, 4, 8, 16, 32, 64]
    batch_sizes['RM2'] = [1, 2, 4, 8, 16, 32]
    batch_sizes['RM3'] = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes['RM4'] = [1, 2, 4, 8, 16]
    batch_sizes['kaggle'] = [1, 2, 4, 8, 16, 32, 64]
    zipf_parameters = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    clean_env()
    
    for model in models:
        
        print("----------------------------------------------------------------")
        print("current model: " + model)
        print("----------------------------------------------------------------")
        
        emb_pool_process_id = start_emb_pool(model)
        
        for batch_size in batch_sizes[model]:
            start_coordinator(model, batch_size, 1.5)
            
        # for zipf_parameter in zipf_parameters:
        #     start_coordinator(model, 32, zipf_parameter)
            
        close_emb_pool()
            

if __name__ == "__main__":
    run()