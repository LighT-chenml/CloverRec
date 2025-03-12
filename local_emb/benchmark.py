import subprocess
import argparse
import asyncio

def start_coordinator(model, batch_size):
    command = f"./benchmark_coordinator_{model}.sh {batch_size}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

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
    batch_sizes['RM1'] = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes['RM2'] = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes['RM3'] = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes['RM4'] = [1, 2, 4, 8, 16, 32, 64]
    batch_sizes['kaggle'] = [1, 2, 4, 8, 16, 32, 64]
    
    clean_env()
    
    for model in models:
        
        print("----------------------------------------------------------------")
        print("current model: " + model)
        print("----------------------------------------------------------------")
        
        for batch_size in batch_sizes[model]:
            start_coordinator(model, batch_size)

if __name__ == "__main__":
    run()