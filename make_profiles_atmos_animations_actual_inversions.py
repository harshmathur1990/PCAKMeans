import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')

output_atmos_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1/frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_atmos.nc')

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-6, -5])

x = [662, 712]

y = [708, 758]


def plot_fov_parameter_variation(
    animation_path,
    wave_indices,
    tau_indices,
    fps=1
):

    global calib_velocity

    sys.stdout.write('Animation Path: {}\n'.format(animation_path))
    sys.stdout.write('wave_indices: {}\n'.format(wave_indices))
    sys.stdout.write('Tau Indices: {}\n'.format(tau_indices))

    plt.cla()

    plt.clf()

    plt.close('all')

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    sys.stdout.write('Read Spectra File\n')

    image0 = np.mean(data[0, 0, wave_indices, x[0]:x[1], y[0]:y[1]], axis=0)

    sys.stdout.write('Made Image\n')

    calib_velocity = 333390.00079943583
    # if calib_velocity is None:
    #     calib_velocity = get_calib_velocity()

    # print (calib_velocity)
    # calib_velocity = 357188.5568518038

    sys.stdout.write('Calib Velocity: {}\n'.format(calib_velocity))

    f = h5py.File(output_atmos_filepath, 'r')

    atmos_indices = np.where(
        (f['ltau500'][0, 0, 0] >= tau_indices[0]) &
        (f['ltau500'][0, 0, 0] <= tau_indices[1])
    )[0]

    temp0 = np.mean(f['temp'][0, :, :, atmos_indices], axis=2)

    vlos0 = np.mean(f['vlos'][0, :, :, atmos_indices] - calib_velocity, axis=2) / 1e5

    vturb0 = np.mean(f['vturb'][0, :, :, atmos_indices], axis=2) / 1e5

    fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=100, gridspec_kw={'wspace': 0.001, 'hspace': 0.025})

    vlos_levels = np.array(
        list(np.linspace(f['vlos'][()].min(), 0, 20)) +
        list(np.linspace(0, f['vlos'][()].max(), 20))[1:]
    ) / 1e5
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
                data[j, 0, wave_indices, x[0]:x[1], y[0]:y[1]],
                axis=0
            )
        )
        im1.set_array(
            np.mean(f['temp'][j, :, :, atmos_indices], axis=2)
        )
        im2.set_array(
            np.mean(f['vlos'][j, :, :, atmos_indices] - calib_velocity, axis=2) / 1e5
        )
        im3.set_array(
            np.mean(f['vturb'][j, :, :, atmos_indices], axis=2) / 1e5
        )

        fig.suptitle('Frame {}'.format(j))
        # return the artists set
        return [im0, im1, im2, im3]

    rate = 1000 / fps

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(21),
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

    calib_velocity = None

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
