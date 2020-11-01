import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

spectra_file_path = Path('/data/harsh1/colabd/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
# spectra_file_path = Path('/home/harsh/Harsh9599771751/colabd/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
label_file_path = Path('/data/harsh1/out_45.h5')
# label_file_path = Path('/home/harsh/Harsh9599771751/Oslo Work/out_45.h5')
rp_path = Path('/data/harsh1/accepted_rp_inversions')
# rp_path = Path('/home/harsh/OsloAnalysis/accepted_rp_inversions')

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-6, -5])


def get_atmos_values_for_lables(tau_indices):

    temp = np.zeros(45)

    vlos = np.zeros(45)

    vturb = np.zeros(45)

    for i in range(45):
        rp_dir_path = rp_path / 'rp_{}'.format(i)

        all_files = rp_dir_path.glob('**/*')
        atmos_file = [
            x for x in all_files if x.is_file() and x.name.endswith('.nc')
        ]
        atmos_file = atmos_file[0]
        f = h5py.File(atmos_file, 'r')

        atmos_indices = np.where(
            (f['ltau500'][0, 0, 0] >= tau_indices[0]) &
            (f['ltau500'][0, 0, 0] <= tau_indices[1])
        )[0]

        temp[i] = np.mean(f['temp'][0, 0, 0, atmos_indices])

        vlos[i] = np.mean(f['vlos'][0, 0, 0, atmos_indices])

        vturb[i] = np.mean(f['vturb'][0, 0, 0, atmos_indices])

        f.close()

    return temp, vlos, vturb


def prepare_get_value_at_pixel(label_map, value_map):

    def get_value_at_pixel(i, j):
        return value_map[label_map[int(i), int(j)]]

    return get_value_at_pixel


def get_image_map(label_map, value_map):

    get_value_at_pixel = prepare_get_value_at_pixel(label_map, value_map)

    vec_get_value_at_pixel = np.vectorize(get_value_at_pixel)

    return np.fromfunction(
        vec_get_value_at_pixel,
        shape=label_map.shape,
        dtype=np.float64
    )


def get_calib_velocity():
    
    vlos = np.zeros(45)

    for i in range(45):
        rp_dir_path = rp_path / 'rp_{}'.format(i)

        all_files = rp_dir_path.glob('**/*')
        atmos_file = [
            x for x in all_files if x.is_file() and
            x.name.endswith('.nc')
        ][0]

        f = h5py.File(atmos_file, 'r')

        atmos_indices = np.where(
            (f['ltau500'][0, 0, 0] >= photosphere_tau[0]) &
            (f['ltau500'][0, 0, 0] <= photosphere_tau[1])
        )[0]

        vlos[i] = np.mean(f['vlos'][0, 0, 0, atmos_indices])

        f.close()

    f = h5py.File(label_file_path, 'r')

    weights = np.ones(45)

    for i in range(45):
        a, b, c = np.where(f['final_labels'][()] == i)
        weights[i] = a.shape[0]

    f.close()

    return np.sum(vlos * weights) / np.sum(weights)


def plot_fov_parameter_variation(
    animation_path,
    wave_indices,
    tau_indices,
    fps=1
):

    sys.stdout.write('Animation Path: {}\n'.format(animation_path))
    sys.stdout.write('wave_indices: {}\n'.format(wave_indices))
    sys.stdout.write('Tau Indices: {}\n'.format(tau_indices))

    plt.cla()

    plt.clf()

    plt.close('all')

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    sys.stdout.write('Read Spectra File\n')

    image0 = np.mean(data[0, 0, wave_indices], axis=0)

    sys.stdout.write('Made Image\n')

    calib_velocity = get_calib_velocity()

    sys.stdout.write('Calib Velocity: {}\n'.format(calib_velocity))

    temp, vlos, vturb = get_atmos_values_for_lables(tau_indices)

    sys.stdout.write('Calculated Atmos Params\n')

    vlos -= calib_velocity

    vlos /= 1e5

    vturb /= 1e5

    f = h5py.File(label_file_path, 'r')

    labels = f['final_labels']

    temp0 = get_image_map(labels[0], temp)

    vlos0 = get_image_map(labels[0], vlos)

    vturb0 = get_image_map(labels[0], vturb)

    fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=100, gridspec_kw={'wspace': 0.001, 'hspace': 0.025})

    vlos_levels = np.array(
        list(np.linspace(vlos.min(), 0, 20)) +
        list(np.linspace(0, vlos.max(), 20))[1:]
    )
    vlos_cmap = plt.cm.get_cmap('coolwarm', vlos_levels.shape[0])
    vlos_norm = matplotlib.colors.BoundaryNorm(vlos_levels, vlos_cmap.N)

    im0 = axs[0][0].imshow(
        image0,
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    im1 = axs[0][1].imshow(
        temp0,
        origin='lower',
        cmap='hot',
        interpolation='none'
    )

    im2 = axs[1][0].imshow(
        vlos0,
        origin='lower',
        cmap=vlos_cmap,
        interpolation='none',
        norm=vlos_norm
    )

    im3 = axs[1][1].imshow(
        vturb0,
        origin='lower',
        cmap='copper',
        interpolation='none'
    )

    axs[0][0].set_xticklabels([])
    axs[0][0].set_yticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][0].set_yticklabels([])
    axs[1][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])

    axins0 = inset_axes(
        axs[0][0],
        width="5%",
        height="50%",
        loc='lower left',
        bbox_to_anchor=(-0.25, 0., 1, 1),
        bbox_transform=axs[0][0].transAxes,
        borderpad=0,
    )

    axins1 = inset_axes(
        axs[0][1],
        width="5%",
        height="50%",
        loc='lower right',
        bbox_to_anchor=(0.15, 0., 1, 1),
        bbox_transform=axs[0][1].transAxes,
        borderpad=0,
    )

    axins2 = inset_axes(
        axs[1][0],
        width="5%",
        height="50%",
        loc='lower left',
        bbox_to_anchor=(-0.25, 0., 1, 1),
        bbox_transform=axs[1][0].transAxes,
        borderpad=0,
    )

    axins3 = inset_axes(
        axs[1][1],
        width="5%",
        height="50%",
        loc='lower right',
        bbox_to_anchor=(0.15, 0., 1, 1),
        bbox_transform=axs[1][1].transAxes,
        borderpad=0,
    )

    cbar0 = fig.colorbar(im0, cax=axins0)
    cbar1 = fig.colorbar(im1, cax=axins1)
    cbar2 = fig.colorbar(
        im2,
        cax=axins2,
        spacing='uniform',
        extend='both',
        extendfrac='auto',
        extendrect='True',
        boundaries=vlos_levels
    )
    cbar3 = fig.colorbar(im3, cax=axins3)

    cbar0.ax.tick_params(labelsize=10)
    cbar1.ax.tick_params(labelsize=10)
    cbar2.ax.tick_params(labelsize=10)
    cbar3.ax.tick_params(labelsize=10)

    # fig.tight_layout()

    def updatefig(j):
        # set the data in the axesimage object
        im0.set_array(
            np.mean(
                data[j, 0, wave_indices],
                axis=0
            )
        )
        im1.set_array(
            get_image_map(labels[j], temp)
        )
        im2.set_array(
            get_image_map(labels[j], vlos)
        )
        im3.set_array(
            get_image_map(labels[j], vturb)
        )

        fig.suptitle('Frame {}'.format(j))
        # return the artists set
        return [im0, im1, im2, im3]

    rate = 1000 / fps

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(100),
        interval=rate,
        blit=True
    )

    Writer = animation.writers['ffmpeg']

    writer = Writer(
        fps=fps,
        metadata=dict(artist='Harsh Mathur'),
        bitrate=1800
    )

    ani.save(animation_path, writer=writer)

    f.close()

    plt.cla()

    plt.close(fig)

    plt.close('all')


if __name__ == '__main__':

    plot_fov_parameter_variation(
        animation_path='photosphere_map.mp4',
        wave_indices=photosphere_indices,
        tau_indices=photosphere_tau
    )

    plot_fov_parameter_variation(
        animation_path='mid_chromosphere_map.mp4',
        wave_indices=mid_chromosphere_indices,
        tau_indices=mid_chromosphere_tau
    )

    plot_fov_parameter_variation(
        animation_path='upper_chromosphere_map.mp4',
        wave_indices=upper_chromosphere_indices,
        tau_indices=upper_chromosphere_tau
    )
