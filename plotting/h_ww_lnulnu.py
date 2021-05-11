import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sys
import os
import gc

sys.path.append(".")
from data_modules.h_ww_lnulnu import HWWLNuLNuDataModule
from modules.h_ww_lnulnu import HWWLNuLNuModule

def inference():
    dm = HWWLNuLNuDataModule(batch_size=825000)
    val_loader = dm.val_dataloader()

    lm = HWWLNuLNuModule.load_from_checkpoint(
        "lightning_logs/version_2/checkpoints/epoch=13-step=6355.ckpt")

    for (batch, all_data) in val_loader:
        x = torch.cat((batch["La"], batch["Lb"], batch["MET"]), dim=1)
        y_hat = lm.model(x)

        model_df = pd.DataFrame(data=y_hat.detach().numpy(),
                                columns=["Nax", "Nay", "Naz", "Nbz"])
        df = pd.DataFrame(data=all_data.detach().numpy(), columns=['H_Mass', 'H_Genx', 'H_Geny', 'H_Genz', 'H_GenE', 'H_Genm', 'H_Genm_squared', 'Wa_Genx', 'Wa_Geny', 'Wa_Genz', 'Wa_GenE', 'Wa_Genm', 'Wa_Genm_squared', 'Wb_Genx', 'Wb_Geny', 'Wb_Genz', 'Wb_GenE', 'Wb_Genm', 'Wb_Genm_squared', 'La_Visx', 'La_Visy', 'La_Visz', 'La_VisE', 'La_Vism', 'La_Vism_squared', 'Na_Genx', 'Na_Geny', 'Na_Genz', 'Na_GenE', 'Na_Genm', 'Na_Genm_squared', 'Lb_Visx', 'Lb_Visy', 'Lb_Visz', 'Lb_VisE', 'Lb_Vism', 'Lb_Vism_squared', 'Nb_Genx', 'Nb_Geny', 'Nb_Genz', 'Nb_GenE', 'Nb_Genm', 'Nb_Genm_squared', 'MET_X_Vis', 'MET_Y_Vis', 'Hx', 'Hy', 'Hz', 'HE', 'Hm', 'Hm_squared', 'Wax', 'Way', 'Waz', 'WaE', 'Wam', 'Wam_squared', 'Wbx', 'Wby', 'Wbz', 'WbE', 'Wbm', 'Wbm_squared', 'Nax', 'Nay', 'Naz', 'NaE', 'Nam', 'Nam_squared', 'Nbx', 'Nby', 'Nbz', 'NbE', 'Nbm', 'Nbm_squared'])

        return model_df, df


def compute_model_results():
    results_path = "data/h_ww_lnlnu_results.pkl"
    val_set_path = "data/h_ww_lnlnu_val.pkl"

    if os.path.exists(results_path):
        model_df = pd.read_pickle(results_path)
        df = pd.read_pickle(val_set_path)
    else:
        model_df, df = inference()

        # compute energy

        model_df["Nbx"] = df["MET_X_Vis"] - df["La_Visx"] - df["Lb_Visx"] - model_df["Nax"]
        model_df["Nby"] = df["MET_Y_Vis"] - df["La_Visy"] - df["Lb_Visy"] - model_df["Nay"]

        model_df["Wax"] = df["La_Visx"] + model_df["Nax"]
        model_df["Way"] = df["La_Visy"] + model_df["Nay"]
        model_df["Waz"] = df["La_Visz"] + model_df["Naz"]

        model_df["Wbx"] = df["Lb_Visx"] + model_df["Nbx"]
        model_df["Wby"] = df["Lb_Visy"] + model_df["Nby"]
        model_df["Wbz"] = df["Lb_Visz"] + model_df["Nbz"]

        model_df["Hx"] = model_df["Wax"] + model_df["Wbx"]
        model_df["Hy"] = model_df["Way"] + model_df["Wby"]
        model_df["Hz"] = model_df["Waz"] + model_df["Wbz"]

        # compute energy
        model_df["NaE"] = (model_df["Nax"] ** 2 + model_df["Nay"] ** 2 + model_df["Naz"] ** 2) ** 0.5
        model_df["NbE"] = (model_df["Nbx"] ** 2 + model_df["Nby"] ** 2 + model_df["Nbz"] ** 2) ** 0.5
        model_df["WaE"] = model_df["NaE"] + df["La_VisE"]
        model_df["WbE"] = model_df["NbE"] + df["Lb_VisE"]
        model_df["HE"] = model_df["WaE"] + model_df["WbE"]

        # compute mass
        model_df["Wam"] = (model_df["WaE"] ** 2 - (model_df["Wax"] ** 2 + model_df["Way"] ** 2 + model_df["Waz"] ** 2)) ** 0.5
        model_df["Wbm"] = (model_df["WbE"] ** 2 - (model_df["Wbx"] ** 2 + model_df["Wby"] ** 2 + model_df["Wbz"] ** 2)) ** 0.5
        model_df["Hm"] = (model_df["HE"] ** 2 - (model_df["Hx"] ** 2 + model_df["Hy"] ** 2 + model_df["Hz"] ** 2)) ** 0.5

        model_df.to_pickle(results_path)
        df.to_pickle(val_set_path)

    return model_df, df


def fig1(df):
    df_500 = df.loc[df["H_Mass"] == 500]

    norm_Hm = df_500["Hm"] / df_500["H_Genm"]
    norm_Wam = df_500["Wam"] / df_500["Wa_Genm"]

    x_name = "Hm_rjr / Hm_truth"
    y_name = "Wam_rjr / Wam_truth"

    plot_df = pd.DataFrame(data={x_name: norm_Hm, y_name: norm_Wam})

    fg = sns.displot(plot_df, x=x_name, y=y_name)
    fg.ax.set_xlim(0.2, 1.8)
    fg.ax.set_ylim(0, 1.8)
    plt.savefig("figures/fig1.png")


def fig2(df, model_df):
    # (gen key into df, key for rjr in df and model_df)
    field_names = list(model_df.columns)

    def plot_dis(key, gen_key, binrange, nbins=50):
        save_file = f"figures/fig2_{key}.png"
        if os.path.exists(save_file):
            return

        print(f"Plotting {save_file}")
        sns.histplot(
            {
                f"{key}_model": model_df[key],
                gen_key: df[gen_key],
                f"{key}_rjr": df[key]
            },
            binwidth=(binrange[1] - binrange[0]) / nbins,
            binrange=binrange,
            alpha=0.5)

        plt.savefig(save_file)
        plt.clf()
        gc.collect()

    plot_dis("Nax", "Na_Genx", (-500, 500))
    plot_dis("Nay", "Na_Geny", (-500, 500))
    plot_dis("Naz", "Na_Genz", (-1000, 1000))

    plot_dis("Nbx", "Na_Genx", (-750, 750))
    plot_dis("Nby", "Na_Geny", (-750, 750))
    plot_dis("Nbz", "Na_Genz", (-2000, 2000))

    plot_dis("Wax", "Wa_Genx", (-750, 750))
    plot_dis("Way", "Wa_Geny", (-750, 750))
    plot_dis("Waz", "Wa_Genz", (-2000, 2000))

    plot_dis("Wbx", "Wb_Genx", (-750, 750))
    plot_dis("Wby", "Wb_Geny", (-750, 750))
    plot_dis("Wbz", "Wb_Genz", (-2000, 2000))

    plot_dis("Hx", "Wb_Genx", (-500, 500))
    plot_dis("Hy", "Wb_Geny", (-500, 500))
    plot_dis("Hz", "Wb_Genz", (-3500, 3500))

    plot_dis("NaE", "Na_GenE", (0, 1500))
    plot_dis("NbE", "Nb_GenE", (0, 1500))
    plot_dis("WaE", "Wa_GenE", (0, 3000))
    plot_dis("WbE", "Wb_GenE", (0, 3000))
    plot_dis("HE", "H_GenE", (0, 5000))

    plot_dis("Wam", "Wa_Genm", (0, 250))
    plot_dis("Wbm", "Wb_Genm", (0, 250))
    plot_dis("Hm", "H_Genm", (0, 2000))


def main():
    model_df, df = compute_model_results()
    print(f"Dataframe colums: {df.columns}")
    gc.collect()

    # fig1(df)
    fig2(df, model_df)


if __name__ == "__main__":
    main()
