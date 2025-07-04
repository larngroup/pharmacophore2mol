import os
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from diffusers import DDPMScheduler
import torch



class NoisySubGridsDataset(SubGridsDataset):
    def __init__(self, schedule_type="cosine", n_timesteps=1000, *args, **kwargs):
        """
        Dataset for 3D pharmacophore and molecule voxel grid fragments with added noise.

        Parameters
        ----------
        schedule_type : str
            Type of noise schedule to use. Options are "cosine" or "linear".
        *args, **kwargs : 
            Additional arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.n_timesteps = n_timesteps

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
        mol_frag: torch.Tensor
            Noised voxel grid of the molecule fragment.
        timestep: torch.Tensor
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
    dataset = NoisySubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=100, transforms=[])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)

    for pharm_frag, noised_mol_frag, timestep in dataloader:
        print(f"Pharmacophore fragment shape: {pharm_frag.shape}")
        print(f"Noised molecule fragment shape: {noised_mol_frag.shape}")
        print(f"Timestep: {timestep}")
        


