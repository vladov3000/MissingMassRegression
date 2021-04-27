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
        "lightning_logs/version_2/checkpoints/epoch=0-step=453.ckpt")

    for batch in val_loader:
        x = torch.cat((batch["La"], batch["Lb"], batch["MET"]), dim=1)
        y_hat = lm.model(x)

        model_df = pd.DataFrame(data=y_hat.detach().numpy(),
                                columns=["Nax", "Nay", "Naz", "Nbz"])

        return model_df


def compute_model_results(df):
    results_path = "data/h_ww_lnlnu_results.pkl"

    if os.path.exists(results_path):
        model_df = pd.read_pickle(results_path)
    else:
        model_df = inference()

        # compute energy

        model_df["Nbx"] = df["MET_X_Vis"] - df["La_Visx"] - df[
            "Lb_Visx"] - model_df["Nax"]
        model_df["Nby"] = df["MET_Y_Vis"] - df["La_Visy"] - df[
            "Lb_Visy"] - model_df["Nay"]

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
        model_df["Wam"] = (model_df["WaE"] - (model_df["Wax"] ** 2 + model_df["Way"] ** 2 + model_df["Waz"] ** 2) ** 0.5) ** 0.5
        model_df["Wbm"] = (model_df["WbE"] - (model_df["Wbx"] ** 2 + model_df["Wby"] ** 2 + model_df["Wbz"] ** 2) ** 0.5) ** 0.5
        model_df["Hm"] = (model_df["HE"] - (model_df["Hx"] ** 2 + model_df["Hy"] ** 2 + model_df["Hz"] ** 2) ** 0.5) ** 0.5

        model_df.to_pickle(results_path)

    print(model_df)

    return model_df


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

    for field in field_names:
        key = field
        gen_key = f"{field[:-1]}_Gen{field[-1]}"

        save_file = f"figures/fig2_{key}.png"
        if os.path.exists(save_file):
            continue
        print(f"Plotting {save_file}")

        sns.histplot(
            {
                f"{key}_model": model_df[key],
                gen_key: df[gen_key],
                f"{key}_rjr": df[key]
            },
            alpha=0.5)

        plt.savefig(save_file)

        save_file = f"figures/fig2_norm_{key}.png"
        if os.path.exists(save_file):
            continue
        print(f"Plotting {save_file}")

        sns.histplot(
            {
                f"{key}_model / truth": model_df[key] / df[gen_key],
                f"{key}_rjr / truth": df[key] / df[gen_key],
            },
            alpha=0.5)

        plt.savefig()


def main():
    df = pd.read_pickle("data/h_ww_lnulnu.pkl")
    model_df = compute_model_results(df)
    print(f"Dataframe colums: {df.columns}")

    # fig1(df)
    fig2(df, model_df)


if __name__ == "__main__":
    main()
