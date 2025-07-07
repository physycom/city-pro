import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt




def Plot_fraction_of_points_in_other_classe(DfScatter,PlotDir,SaveName,SaveFig = False):
    len_classes = len(DfScatter["class"].unique())
    matrix_exchange = np.zeros((len_classes,len_classes))
    for starting_class,df_class in DfScatter.groupby("class"):
        for class_arrival in DfScatter["class"].unique():
            matrix_exchange[int(starting_class),int(class_arrival)] = np.mean(df_class[class_arrival])
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(matrix_exchange, cmap="viridis")

    # Add a color bar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Value")

    # Add annotations (values in the matrix)
    for i in range(matrix_exchange.shape[0]):
        for j in range(matrix_exchange.shape[1]):
            ax.text(j, i, f"{matrix_exchange[i, j]:.2f}", ha="center", va="center", color="white" if matrix_exchange[i, j] < 0.5 else "black")

    # Set axis labels
    ax.set_xlabel("Class Arrival")
    ax.set_ylabel("Class Starting")

    # Set tick labels
    num_classes = matrix_exchange.shape[0]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels([f"Class {i}" for i in range(num_classes)])
    ax.set_yticklabels([f"Class {i}" for i in range(num_classes)])
    if SaveFig:
        plt.savefig(os.path.join(PlotDir,SaveName))
