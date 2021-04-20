import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def fig11():
    df = pd.read_pickle("data/h_ww_lnulnu.pkl")
    df_500 = df.loc[df["H_Mass"] == 500]
    df_500 = df_500[["Hm", "Wa_Genm"]]
    sns.displot(df_500, x="Hm", y="Wa_Genm")
    plt.show()


def main():
    fig11()


if __name__ == "__main__":
    main()