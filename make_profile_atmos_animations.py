import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# base_path = Path('/home/harsh/OsloAnalysis/new_kmeans')

base_path = Path('/home/harsh/animation_data')
spectra_file_path = Path('/data/harsh1/colabd/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
# spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
label_file_path = base_path / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
# label_file_path = Path('/home/harsh/Harsh9599771751/Oslo Work/out_45.h5')
# rp_path = Path('/data/harsh1/accepted_rp_inversions')
# rp_path = Path('/home/harsh/OsloAnalysis/accepted_rp_inversions')

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-6, -5])

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 8, 44, 63, 84]

shock_spicule_profiles = [4, 10, 19, 26, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 72, 77, 92, 99]

retry_shock_spicule = [6, 49, 18, 36]

shock_78 = [78]

reverse_shock_profiles = [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]


def get_filepath_and_content_list(rp):

    global base_path

    if rp in quiet_profiles:
        filename, content_list = base_path / 'quiet_profiles/plots_v1/rp_0_3_8_11_14_15_16_20_21_24_28_31_34_40_42_43_44_47_48_51_57_58_60_61_62_63_66_69_70_71_73_74_75_76_82_84_86_89_90_99_cycle_1_t_5_vl_1_vt_4_atmos.nc', [0, 3, 8, 11, 14, 15, 16, 20, 21, 24, 28, 31, 34, 40, 42, 43, 44, 47, 48, 51, 57, 58, 60, 61, 62, 63, 66, 69, 70, 71, 73, 74, 75, 76, 82, 84, 86, 89, 90, 99]

    elif rp in shock_spicule_profiles:
        filename, content_list = base_path / 'shock_and_spicule_profiles/plots_v1/rp_1_4_6_10_15_18_19_20_22_23_24_26_36_37_40_43_49_52_53_55_56_62_66_70_72_73_74_75_77_78_79_84_85_86_92_94_99_cycle_1_t_4_vl_5_vt_4_atmos.nc', [1, 4, 6, 10, 15, 18, 19, 20, 22, 23, 24, 26, 36, 37, 40, 43, 49, 52, 53, 55, 56, 62, 66, 70, 72, 73, 74, 75, 77, 78, 79, 84, 85, 86, 92, 94, 99]

    elif rp in retry_shock_spicule:
        filename, content_list = base_path / 'retry_shock_spicule/plots_v1/rp_6_49_18_36_78_cycle_1_t_4_vl_5_vt_4_atmos.nc', [6, 49, 18, 36, 78]

    elif rp in shock_78:
        filename, content_list = base_path / 'rp_78/plots_v3/rp_78_cycle_1_t_4_vl_7_vt_4_atmos.nc', [78]

    elif rp in reverse_shock_profiles:
        filename, content_list = base_path / 'reverse_shock_profiles/plots_v1/rp_3_13_16_17_25_32_33_35_41_58_61_64_68_82_95_97_cycle_1_t_5_vl_5_vt_4_atmos.nc', [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

    elif rp in other_emission_profiles:
        filename, content_list = base_path / 'other_emission_profiles/plots_v1/rp_2_5_7_9_12_27_29_30_38_39_45_46_50_54_57_59_65_67_71_76_80_81_83_87_88_91_93_96_98_cycle_1_t_5_vl_5_vt_4_atmos.nc', [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

    return filename, content_list


def get_atmos_values_for_lables(tau_indices):

    temp = np.zeros(100)

    vlos = np.zeros(100)

    vturb = np.zeros(100)

    for i in range(100):
        filename, content_list = get_filepath_and_content_list(i)

        f = h5py.File(filename, 'r')

        index = content_list.index(i)

        if len(content_list) == f['ltau500'].shape[1]:
            normal = True
        else:
            normal = False

        if normal:
            atmos_indices = np.where(
                (f['ltau500'][0, index, 0] >= tau_indices[0]) &
                (f['ltau500'][0, index, 0] <= tau_indices[1])
            )[0]

            temp[i] = np.mean(f['temp'][0, index, 0, atmos_indices])

            vlos[i] = np.mean(f['vlos'][0, index, 0, atmos_indices])

            vturb[i] = np.mean(f['vturb'][0, index, 0, atmos_indices])

        else:
            atmos_indices = np.where(
                (f['ltau500'][0, 0, index] >= tau_indices[0]) &
                (f['ltau500'][0, 0, index] <= tau_indices[1])
            )[0]

            temp[i] = np.mean(f['temp'][0, 0, index, atmos_indices])

            vlos[i] = np.mean(f['vlos'][0, 0, index, atmos_indices])

            vturb[i] = np.mean(f['vturb'][0, 0, index, atmos_indices])

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

    vlos = np.zeros(100)

    for i in range(100):
        filename, content_list = get_filepath_and_content_list(i)

        f = h5py.File(filename, 'r')

        index = content_list.index(i)

        # print (i, filename, content_list)

        if f['ltau500'].shape[1] == len(content_list):
            normal = True
        else:
            normal = False

        if normal:
            atmos_indices = np.where(
                (f['ltau500'][0, index, 0] >= photosphere_tau[0]) &
                (f['ltau500'][0, index, 0] <= photosphere_tau[1])
            )[0]

            vlos[i] = np.mean(f['vlos'][0, index, 0, atmos_indices])

        else:
            atmos_indices = np.where(
                (f['ltau500'][0, 0, index] >= photosphere_tau[0]) &
                (f['ltau500'][0, 0, index] <= photosphere_tau[1])
            )[0]

            vlos[i] = np.mean(f['vlos'][0, 0, index, atmos_indices])
        f.close()

    f = h5py.File(label_file_path, 'r')

    weights = np.ones(100)

    for i in range(100):
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

    global calib_velocity

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

    if calib_velocity is None:
        calib_velocity = get_calib_velocity()

    print (calib_velocity)
    # calib_velocity = 357188.5568518038

    sys.stdout.write('Calib Velocity: {}\n'.format(calib_velocity))

    temp, vlos, vturb = get_atmos_values_for_lables(tau_indices)

    sys.stdout.write('Calculated Atmos Params\n')

    vlos -= calib_velocity

    vlos /= 1e5

    vturb /= 1e5

    f = h5py.File(label_file_path, 'r')

    labels = f['new_final_labels'][()].astype(np.int64)

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
