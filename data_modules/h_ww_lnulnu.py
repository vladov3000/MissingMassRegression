import os

import torch
from torch.utils.data import DataLoader

import pandas as pd
import pytorch_lightning as pl

DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "data")
DEFAULT_DATA_FILE = os.path.join(DEFAULT_DATA_FOLDER, "h_ww_lnulnu.pkl")
DEFAULT_DATA_SPLIT = [0.9, 0.05, 0.05]


class HWWLNuLNuDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, h_mass=None, h_mass_exclude=None, h_mass_lt=None, h_mass_gt=None):
        print(f"Using HWWLNuLNuDataset with dataframe file at: {df_path}"
              + f"with H_mass = {h_mass}, H_mass_exclude = {h_mass_exclude}, H_mass_lt = {h_mass_lt}, H_mass_gt = {h_mass_gt}")
        df = pd.read_pickle(df_path)
        if h_mass is not None:
            df = df[df["H_Mass"] == h_mass]
        if h_mass_exclude is not None:
            df = df[df["H_Mass"] != h_mass_exclude]
        if h_mass_lt is not None:
            df = df[df["H_Mass"] < h_mass_lt]
        if h_mass_gt is not None:
            df = df[df["H_Mass"] > h_mass_gt]

        print(f"Data column names: {df.columns}")
        self.col_names = list(df.columns)
        self.col_names_idx = {i: v for v, i in enumerate(list(df.columns))}
        self.data = torch.from_numpy(df.to_numpy())

    def get_vec(self, idx, names):
        return self.data[idx][[self.col_names_idx[i] for i in names]]

    def __getitem__(self, idx):
        return {
                   "La": self.get_vec(idx, ("La_Visx", "La_Visy", "La_Visz")),
                   "Lb": self.get_vec(idx, ("Lb_Visx", "Lb_Visy", "Lb_Visz")),
                   "Na": self.get_vec(idx, ("Na_Genx", "Na_Geny", "Na_Genz")),
                   "Nb": self.get_vec(idx, ("Nb_Genx", "Nb_Geny", "Nb_Genz")),
                   "Nbz": self.get_vec(idx, ["Nb_Genz"]),
                   "MET": self.get_vec(idx, ("MET_X_Vis", "MET_Y_Vis"))
               }, self.data[idx]

    def __len__(self):
        return len(self.data)


class HWWLNuLNuDataModule(pl.LightningDataModule):
    def __init__(self,
                 df_path=DEFAULT_DATA_FILE,
                 data_split=None,
                 batch_size=1048576,
                 num_workers=8,
                 h_mass=None,
                 h_mass_exclude=None,
                 h_mass_lt=None,
                 h_mass_gt=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        print("Using HWWDataModule.")

        if data_split is None:
            data_split = DEFAULT_DATA_SPLIT

        self.batch_size = batch_size
        self.num_workers = num_workers

        dataset = HWWLNuLNuDataset(df_path, h_mass=h_mass, h_mass_exclude=h_mass_exclude, h_mass_lt=h_mass_lt,
                                   h_mass_gt=h_mass_gt)
        data_split_nums = [int(i * len(dataset)) for i in data_split]
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            dataset, lengths=data_split_nums)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
