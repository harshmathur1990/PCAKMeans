import h5py
import numpy as np
import sunpy.io
import matplotlib.pyplot as plt


def contrast_variation(filename, figsize=None, dpi=None):

    data, header = sunpy.io.read_file(filename)[0]

    contrast = np.zeros(shape=(data.shape[0], ))

    for i in np.arange(contrast.size):
        contrast[i] = data[i][0][0].std() / data[i][0][0].mean()

    f = h5py.File('contrast_variation.h5', 'w')

    f['contrast'] = contrast

    f.close()

    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax1 = fig.add_subplot(111)

    ax1.bar(np.arange(contrast.size), contrast, color='black')

    plt.xticks(np.arange(contrast.size), rotation='vertical')

    plt.title('Contrast Variation')

    plt.ylabel('Contrast')

    plt.xlabel('Frame Number')

    plt.legend()

    plt.tight_layout()

    plt.savefig('contrast_variation.png')

    plt.clf()

    plt.cla()
