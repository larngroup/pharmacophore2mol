import os
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from diffusers import DDPMScheduler
import torch



class NoisySubGridsDataset(SubGridsDataset):
    def __init__(self, schedule_type="cosine", n_timesteps=1000, return_clean=False, *args, **kwargs):
        """
        Dataset for 3D pharmacophore and molecule voxel grid fragments with added noise.

        Parameters
        ----------
        schedule_type : str
            Type of noise schedule to use. Options are "cosine" or "linear".
        n_timesteps : int
            Number of timesteps for the noise schedule. Default is 1000.
        return_clean : bool
            If True, the dataset will return the clean molecule fragment along with the noised one. Default is False. See "Returns" section for details.
        *args, **kwargs : 
            Additional arguments passed to the parent class.


        Returns
        -------
        pharm_frag: torch.Tensor
            Voxel grid of the pharmacophore fragment (no noise).
        mol_frag: torch.Tensor, Optional
            Voxel grid of the molecule fragment (no noise). Only poresent if `return_clean` is True.
        noisy_mol_frag: torch.Tensor
            Noised voxel grid of the molecule fragment.
        added_noise: torch.Tensor
            Noise that was added to the molecule fragment.
        timestep: int
            Random timestep that was used for noise addition.
        """
        super().__init__(*args, **kwargs)
        self.n_timesteps = n_timesteps
        self.return_clean = return_clean
        if schedule_type == "cosine":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=n_timesteps,
                beta_schedule="squaredcos_cap_v2",  # Cosine schedule
                prediction_type="epsilon"
            )
        elif schedule_type == "linear":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=n_timesteps,
                beta_schedule="linear",
                prediction_type="epsilon"
            )

    def __getitem__(self, idx):
        """
        Get a noisy sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        pharm_frag: torch.Tensor
            Voxel grid of the pharmacophore fragment (no noise).
        mol_frag: torch.Tensor, Optional
            Voxel grid of the molecule fragment (no noise). Only poresent if `return_clean` is True.
        noisy_mol_frag: torch.Tensor
            Noised voxel grid of the molecule fragment.
        added_noise: torch.Tensor
            Noise that was added to the molecule fragment.
        timestep: int
            Random timestep that was used for noise addition.

        """
        if isinstance(idx, slice):
            idxs = range(*idx.indices(self.len))
            return [self[i] for i in idxs]
        if isinstance(idx, int):
            pharm_frag, mol_frag = super().__getitem__(idx)

            # Randomly select a timestep
            timestep = torch.randint(0, self.n_timesteps, (1,))
            added_noise = torch.randn_like(mol_frag)
            # Add noise to the molecule fragment
            noised_mol_frag = self.scheduler.add_noise(mol_frag.unsqueeze(0), added_noise.unsqueeze(0), timestep).squeeze(0)
            added_noise = noised_mol_frag - mol_frag #@maryam this fixes that added noise not being scaled bug. i know that, performance wise, it may be slower, but this is cpu work so no concerns for now
            if self.return_clean:
                # If return_clean is True, return the clean molecule fragment as well
                return pharm_frag, mol_frag, noised_mol_frag, added_noise, timestep.item()
            return pharm_frag, noised_mol_frag, added_noise, timestep.item()
    
    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return super().__len__()
    


if __name__ == "__main__":
    # Example usage
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    from datetime import datetime as dt
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"./runs/dataset_test_at_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    dataset = NoisySubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=100, transforms=[])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)

    for pharm_frag, noised_mol_frag, added_noise, timestep in dataloader:
        print(f"Pharmacophore fragment shape: {pharm_frag.shape}")
        print(f"Noised molecule fragment shape: {noised_mol_frag.shape}")
        print(f"Timestep: {timestep}")
        writer.add_images("Noised Molecule Fragments", noised_mol_frag, global_step=0)
        break
        


