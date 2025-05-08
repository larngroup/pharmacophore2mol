import glob
import multiprocessing
import subprocess
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import atexit
from pharmacophore2mol.models.unet3d.model import UNet3d
from pharmacophore2mol.models.unet3d.config import config
# from pharmacophore2mol.data.utils import worker_tracer_init_fn
# import viztracer


if __name__ == "__main__":
    # multiprocessing.freeze_support()  # For Windows compatibility
    # tracer = viztracer.VizTracer(output_file="main_process_trace.json")
    # tracer.start()
    # print("Main process initialized with tracer")

    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    # Define the dataset and dataloader
    dataset = SubGridsDataset(mols_filepath="../../data/raw/zinc3d_test.sdf")
    # from pympler import asizeof
    # print(len(dataset.cumsum_index))
    # print("Dataset size:", asizeof.asizeof(dataset.cumsum_index), "bytes")

    # import torch.multiprocessing as tmp
    # import multiprocessing as mp
    # tmp.Process = mp.Process

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)#, worker_init_fn=worker_tracer_init_fn) #idk why but calls to __getitem__ from dataloader seem some ms slower than direct calls to __getitem__ from dataset, for the same indexes. even for slices. this is just about the __getitem__ call time, checked by profiling, and not about all other extra methods. TODO: investigate this further.
    # model = UNet3d(in_channels=5, out_channels=8, features=[32, 64, 128, 256]).to(config["device"])
    for epoch in range(config["epochs"]):
        counter = 0
        # Training loop
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc=f"Epoch {epoch + 1}/{config['epochs']}", unit="batch")
        for batch_idx, (data, targets) in loop:
            # exit()
            pass
            if counter == 2:
                break
            counter += 1
            # data = data.to(config["device"])
            # targets = targets.to(config["device"])
            # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            # optimizer.zero_grad()
            # predictions = model(data)
            # with torch.amp.autocast(device_type=config["device"]):
            #     predictions = model(data)
            #     loss = torch.nn.functional.mse_loss(predictions, targets)
            # loss.backward()
            # optimizer.step()

            # # Update the progress bar
            # loop.set_postfix(loss=loss.item())
        break    
    loop.close()
    print("Training complete.")

    # tracer.stop()
    # os.chdir(os.path.join(os.path.dirname(__file__), "../../.."))
    # tracer.save()
    # worker_traces = glob.glob("worker_*_trace.json")
    # subprocess.run(["viztracer", "--combine", "main_process_trace.json"] + worker_traces + ["--output_file", "result.json"])
    # print("Traces combined into result.json")
