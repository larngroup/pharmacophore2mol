import time
from torch.utils.tensorboard import SummaryWriter
import os
# from monai import visualize
import torch
from datetime import datetime
from pharmacophore2mol.models.unet3d_v2.dataset import NoisySubGridsDataset

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    writer = SummaryWriter(log_dir=f"runs/gif_viz_at_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


    dataset = NoisySubGridsDataset(
        mols_filepath="../data/raw/2_pyridone.sdf",
        force_len=1000,
        transforms=[],
        return_clean=True
    )
    dummy_tensor = dataset[0][1].unsqueeze(0)
    #change channel order, CAUSE PYTORCH WILL LIMIT THE CHANNELS TO THE FIRST 3 CHANNELS. if you want to visualize the others you have to either change the order of the channels (for example (C, H, O, N, S) -> (C, H, N, O, S)) or separate into two videos
    dummy_tensor = dummy_tensor[:, [0, 1, 3, 2, 4]]
    print(dummy_tensor.shape)  # Should be (C, D, H, W) or (C, T, H, W) depending on the dataset
    # dummy_tensor = torch.rand(1, 6, 64, 64, 64)  #(BCHWD)
    # with torch's add_video method, we need to reshape the tensor to (B, T, C, H, W) (here T is timestp, so we can use the depth dimention as T)
    dummy_tensor = dummy_tensor.permute(0, 4, 1, 2, 3) 
    print(dummy_tensor.shape)  # Should be (B, T, C, H, W)

    start = time.perf_counter()
    writer.add_video(tag="gif_example", vid_tensor=dummy_tensor, global_step=0, fps=10)
    end = time.perf_counter()
    print(f"Time taken to write video: {end - start:.2f} seconds")
    writer.close()
    # print(type(visualize.img2tensorboard.plot_2d_or_3d_image(
    #     writer=writer,
    #     tag="gif_example",
    #     data=dummy_tensor,  # Remove batch dimension for visualization
    #     step=0,
    #     # max_channels=3,


    #     # scale_factor=255
    #     )))