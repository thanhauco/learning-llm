# 07_distributed_training/intermediate.py
# In this script, we'll implement a hands-on example of distributed data parallelism (DDP)
# using PyTorch's `torch.distributed` package. This is a step up from the theoretical
# concepts in `easy.py`.

# This script is designed to be run from the command line with `torchrun`.
# Example: torchrun --standalone --nproc_per_node=2 07_distributed_training/intermediate.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

print("Intermediate Distributed Training: Hands-on with PyTorch DDP")

def setup(rank, world_size):
    """
    Initializes the distributed process group.
    `torchrun` handles setting the environment variables `MASTER_ADDR` and `MASTER_PORT`.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size) # "gloo" for CPU, "nccl" for GPU
    print(f"Initialized process {rank}/{world_size}")

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    print("Cleaned up process group.")

class ToyModel(nn.Module):
    """A simple model to demonstrate DDP."""
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def run_ddp_example(rank, world_size):
    """
    The main function to run the DDP training example.
    """
    print(f"\n--- Running DDP example on rank {rank}. ---")
    setup(rank, world_size)

    # 1. Create model and move it to the correct device
    # For a real GPU setup, you'd use `device = rank` and move the model to `cuda:{rank}`
    model = ToyModel()
    
    # 2. Wrap the model with DistributedDataParallel
    # This handles gradient synchronization across all processes.
    ddp_model = DDP(model) # For GPU: DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print(f"Rank {rank}: Model wrapped in DDP. Starting training loop.")

    # --- Training Loop Simulation ---
    for i in range(3): # Simulate 3 training steps
        # Create dummy data for this process. In a real scenario, a DistributedSampler
        # would ensure each process gets a unique subset of the data.
        inputs = torch.randn(20, 10) # Dummy input
        labels = torch.randn(20, 5) # Dummy labels

        # Forward pass
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward() # DDP automatically averages gradients across all processes here.

        # Optimizer step
        optimizer.step()

        print(f"Rank {rank}, Step {i}: loss={loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    try:
        # `torchrun` sets these environment variables.
        # We get the rank and world size to pass to our main function.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world_size > 1:
            print(f"World size is {world_size}. Running DDP example.")
            run_ddp_example(rank, world_size)
        else:
            print("World size is 1. DDP requires multiple processes.")
            print("To run this script correctly, use torchrun:")
            print("torchrun --standalone --nproc_per_node=2 07_distributed_training/intermediate.py")

    except ImportError:
        print("\nPyTorch not installed. Skipping this script.")
        print("To run, please install it with: pip install torch")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This script is intended to be run with `torchrun`.")
        print("Example: torchrun --standalone --nproc_per_node=2 07_distributed_training/intermediate.py")

    print("\nIntermediate distributed training script finished.")