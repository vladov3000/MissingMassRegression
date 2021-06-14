import sys

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl


class DenseModel(nn.Module):
    def __init__(self, n_hidden_layers, hidden_dim, loss_name):
        super(DenseModel, self).__init__()
        print("Using DenseModel.")

        def get_block(in_dim, out_dim):
            return (nn.Linear(in_dim, out_dim), nn.ReLU())

        first_layer = get_block(8, hidden_dim)
        hidden_layers = [
            get_block(hidden_dim, hidden_dim) for i in range(n_hidden_layers)
        ]
        out_layer = tuple([nn.Linear(hidden_dim, 4)])

        flattened_layers = list(
            sum([first_layer, *hidden_layers, out_layer], ()))
        self.model = nn.Sequential(*flattened_layers)
        print(f"Model layers: {self.model}")

        if loss_name == "default":
            self.get_loss = self.get_loss_default

    def forward(self, x):
        return self.model(x)

    def get_loss_default(self, batch):
        batch = batch[0]
        x = torch.cat((batch["La"], batch["Lb"], batch["MET"]), dim=1)
        y_hat = self.model(x)
        y = torch.cat((batch["Na"], batch["Nbz"]), dim=1)
        return F.mse_loss(y_hat, y, reduction="sum")

    def get_loss_sim(self, batch):
        batch, all = batch
        x = torch.cat((batch["La"], batch["Lb"], batch["MET"]), dim=1)
        y_hat = self.model(x)

        Na_p = y_hat[0:2]
        Na_Genp = torch.cat((all["Na_Genx"], all["Na_Genx"], all["Na_Genx"]), dim=1)

        return 0


class HWWLNuLNuModule(pl.LightningModule):
    def __init__(self, n_hidden_layers=4, hidden_dim=32, lr=1e-3, loss_name="default"):
        super().__init__()
        print("Using HWWLNuLNuModule.")

        self.lr = lr
        self.save_hyperparameters("n_hidden_layers", "hidden_dim", "lr", "loss_name")

        self.model = DenseModel(n_hidden_layers, hidden_dim, loss_name)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return self.model.get_loss(batch)

    def validation_step(self, batch, batch_idx):
        return self.model.get_loss(batch)

    def test_step(self, batch, batch_idx):
        return self.model.get_loss(batch)
