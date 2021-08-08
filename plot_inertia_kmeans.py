import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


base_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/All Results'
)

k = np.arange(2, 202, 5)

def plot_inertia():
    inertia = list()

    for k_value in k:
        f = h5py.File(
            base_path / 'out_{}.h5'.format(
                k_value
            )
        )

        inertia.append(f['inertia_'][()])
        f.close()

    inertia = np.array(inertia)

    diff_inertia = inertia[:-1]  - inertia[1:]

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(4.135, 4.135,))

    axs.plot(k, inertia / 1e5, color='#364f6B')

    axs.set_xlabel('Number of Clusters')

    axs.set_ylabel(r'$\sigma_{k}\;*\;1e5$')

    axs.grid()

    axs.axvline(x=100, linestyle='--')

    ax2 = axs.twinx()

    ax2.plot(k[1:], diff_inertia / 1e5, color='#3fC1C9')

    ax2.set_ylabel(r'$\sigma_{k} - \sigma_{k+1}\;*\;1e5$')

    axs.yaxis.label.set_color('#364f6B')

    ax2.yaxis.label.set_color('#3fC1C9')

    axs.tick_params(axis='y', colors='#364f6B')

    ax2.tick_params(axis='y', colors='#3fC1C9')

    fig.tight_layout()

    fig.savefig('KMeansInertia.eps', format='eps', dpi=300)
