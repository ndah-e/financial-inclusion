import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def stacked_barplot(df, feature):

    df_grp = pd.DataFrame(
        df.groupby(feature)["bank_account"].value_counts().unstack().fillna(0)
    ).reset_index()
    x = np.array(df_grp.household_size)
    no = np.array(df_grp.No)
    yes = np.array(df_grp.Yes)

    # normalization
    snum = no + yes
    no = no / snum * 100.0
    yes = yes / snum * 100.0

    # stack bars
    plt.figure(figsize=(15, 6))
    plt.bar(x, no, label="No")
    plt.bar(x, yes, bottom=no, label="Yes")

    # add text annotation corresponding to the percentage of each data.
    for xpos, ypos, yval in zip(x, no / 2, no):
        plt.text(xpos, ypos, "%.1f" % yval, ha="center", va="center")

    for xpos, ypos, yval in zip(x, no + yes / 2, yes):
        plt.text(xpos, ypos, "%.1f" % yval, ha="center", va="center")

    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
