import h5py
import sunpy.io
import numpy as np
import matplotlib.pyplot as plt


wave = [
    393.278952, 393.285488, 393.292024, 393.29856, 393.304944,
    93.31148, 393.318016, 393.324552, 393.330936, 393.337472,
    393.344008, 393.350544, 393.356928, 393.363464, 393.37,
    93.376536, 393.383072, 393.389456, 393.395992, 393.402528,
    393.409064, 393.415448, 393.421984, 393.42852, 393.435056,
    393.44144, 393.447976, 393.454512, 393.461048, 400.114744
]



def plot_profiles():

    fig, ax = plt.subplots(5, 9, sharey='col')

    k = 0

    median_profile = np.median(seldata, axis=(0, 2, 3))[:29]

    for i in range(5):
        for j in range(9):
            a, b, c = np.where(labels == k)

            center = np.mean(seldata[a, :, b, c], axis=0)

            ax[i][j].plot(center[:29], color='red')

            ax[i][j].plot(median_profile, color='black')

            ax[i][j].set_ylim(0, 0.000007)

            ax[i][j].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False # labels along the bottom edge are off
            )

            ax[i][j].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False # labels along the bottom edge are off
            )

            k += 1

    plt.show()