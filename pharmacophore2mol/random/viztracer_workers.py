import viztracer
import torch
from torch.utils.data import DataLoader, Dataset
import os
import multiprocessing

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate some work
        return torch.randn(10)

def worker_init_fn(worker_id):
    # Each worker gets its own tracer
    worker_tracer = viztracer.VizTracer(
        output_file=f"worker_{worker_id}_trace.json",
        tracer_entries=1000000
    )
    worker_tracer.start()
    
    # Optional: store the tracer in a global variable so you can access it later
    import builtins
    builtins.__viz_tracer__ = worker_tracer
    
    print(f"Worker {worker_id} initialized with tracer")
    
    # Register an exit handler to save the trace when the worker exits
    import atexit
    def save_trace():
        if hasattr(builtins, "__viz_tracer__"):
            builtins.__viz_tracer__.stop()
            builtins.__viz_tracer__.save()
            print(f"Worker {worker_id} trace saved")
    atexit.register(save_trace)

def main():
    # Main process tracer
    tracer = viztracer.VizTracer(output_file="main_process_trace.json")
    tracer.start()

    # Create dataloader with our custom initialization function
    dataset = SimpleDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        multiprocessing_context='spawn'  # Explicitly set spawn method for Windows
    )

    # Run the dataloader
    for batch in dataloader:
        pass  # Do your processing here

    # Stop the main process tracer
    tracer.stop()
    tracer.save()

    print("Tracing complete. Check the current directory for trace files.")

if __name__ == "__main__":
    # This is crucial for Windows multiprocessing
    multiprocessing.freeze_support()
    main()