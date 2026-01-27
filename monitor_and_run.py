#!/usr/bin/env python3
"""
GPU VRAM Monitor Script
Checks GPU VRAM every 30 seconds and executes grpo.sh when VRAM < 3000MB
"""

import subprocess
import time
import sys

def get_gpu_memory_used():
    """
    Get the current GPU memory usage in MB using nvidia-smi
    Returns the memory used by the first GPU (GPU 0)
    """
    try:
        # Query GPU memory used in MB
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Get first GPU's memory (in case of multiple GPUs)
        memory_used = float(result.stdout.strip().split('\n')[0])
        return memory_used
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error parsing nvidia-smi output: {e}")
        sys.exit(1)

def execute_grpo_script():
    """
    Execute the grpo.sh bash script
    """
    print("\n" + "="*60)
    print("GPU VRAM < 3000MB detected! Executing grpo.sh...")
    print("="*60 + "\n")

    try:
        # Execute the bash script
        subprocess.run(['bash', 'grpo.sh'], check=True)
        print("\n" + "="*60)
        print("grpo.sh execution completed!")
        print("="*60)
    except subprocess.CalledProcessError as e:
        print(f"Error executing grpo.sh: {e}")
        sys.exit(1)

def main():
    print("Starting GPU VRAM monitor...")
    print("Checking every 30 seconds for VRAM < 3000MB")
    print("Press Ctrl+C to stop\n")

    check_count = 0

    try:
        while True:
            check_count += 1
            memory_used = get_gpu_memory_used()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Check #{check_count}: GPU VRAM = {memory_used:.0f} MB")

            if memory_used < 3000:
                execute_grpo_script()
                print("\nMonitoring stopped after script execution.")
                break
            else:
                print(f"  → VRAM still above threshold (need < 3000MB). Waiting 30s...")

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()









