from argparse import ArgumentParser
import sys

import pytorch_lightning as pl
from pytorch_lightning import Trainer

sys.path.append(".")
from data_modules.h_ww_lnulnu import HWWLNuLNuDataModule
from modules.h_ww_lnulnu import HWWLNuLNuModule


def main():
    pl.seed_everything(1234)

    # parse args
    parser = ArgumentParser()

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_hidden_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # define data module
    data_module = HWWLNuLNuDataModule(batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    # define lightning module
    module = HWWLNuLNuModule(n_hidden_layers=args.n_hidden_layers,
                             hidden_dim=args.hidden_dim,
                             lr=args.lr)

    # execute
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()