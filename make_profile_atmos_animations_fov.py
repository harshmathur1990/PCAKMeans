import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


base_path = Path('/home/harsh/OsloAnalysis')

new_kmeans = base_path / 'new_kmeans'

spectra_file_path = base_path / 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'

label_file_path = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-5, -4])

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 8, 44, 63, 84]

shock_spicule_profiles = [4, 10, 19, 26, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 72, 77, 92, 99]

retry_shock_spicule = [6, 49, 18, 36]

shock_78 = [78]

reverse_shock_profiles = [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

x = [662, 712]

y = [708, 758]

frames = [0, 21]

def get_filepath_and_content_list(rp):

    global base_path

    if rp in quiet_profiles:
        filename, content_list = new_kmeans / 'quiet_profiles/plots_v1/rp_0_3_8_11_14_15_16_20_21_24_28_31_34_40_42_43_44_47_48_51_57_58_60_61_62_63_66_69_70_71_73_74_75_76_82_84_86_89_90_99_cycle_1_t_5_vl_1_vt_4_atmos.nc', [0, 3, 8, 11, 14, 15, 16, 20, 21, 24, 28, 31, 34, 40, 42, 43, 44, 47, 48, 51, 57, 58, 60, 61, 62, 63, 66, 69, 70, 71, 73, 74, 75, 76, 82, 84, 86, 89, 90, 99]

    elif rp in shock_spicule_profiles:
        filename, content_list = new_kmeans / 'shock_and_spicule_profiles/plots_v1/rp_1_4_6_10_15_18_19_20_22_23_24_26_36_37_40_43_49_52_53_55_56_62_66_70_72_73_74_75_77_78_79_84_85_86_92_94_99_cycle_1_t_4_vl_5_vt_4_atmos.nc', [1, 4, 6, 10, 15, 18, 19, 20, 22, 23, 24, 26, 36, 37, 40, 43, 49, 52, 53, 55, 56, 62, 66, 70, 72, 73, 74, 75, 77, 78, 79, 84, 85, 86, 92, 94, 99]

    elif rp in retry_shock_spicule:
        filename, content_list = new_kmeans / 'retry_shock_spicule/plots_v1/rp_6_49_18_36_78_cycle_1_t_4_vl_5_vt_4_atmos.nc', [6, 49, 18, 36, 78]

    elif rp in shock_78:
        filename, content_list = new_kmeans / 'rp_78/plots_v3/rp_78_cycle_1_t_4_vl_7_vt_4_atmos.nc', [78]

    elif rp in reverse_shock_profiles:
        filename, content_list = new_kmeans / 'reverse_shock_profiles/plots_v1/rp_3_13_16_17_25_32_33_35_41_58_61_64_68_82_95_97_cycle_1_t_5_vl_5_vt_4_atmos.nc', [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

    elif rp in other_emission_profiles:
        filename, content_list = new_kmeans / 'other_emission_profiles/plots_v1/rp_2_5_7_9_12_27_29_30_38_39_45_46_50_54_57_59_65_67_71_76_80_81_83_87_88_91_93_96_98_cycle_1_t_5_vl_5_vt_4_atmos.nc', [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

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
    wave_indices_list,
    tau_indices_list,
    fps=1
):

    global calib_velocity

    plt.cla()

    plt.clf()

    plt.close('all')

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    sys.stdout.write('Read Spectra File\n')

    image00 = np.mean(
        data[
            0, 0, wave_indices_list[0], x[0]:x[1], y[0]:y[1]
        ],
        axis=0
    )
    image01 = np.mean(
        data[
            0, 0, wave_indices_list[1], x[0]:x[1], y[0]:y[1]
        ],
        axis=0
    )
    image02 = np.mean(
        data[
            0, 0, wave_indices_list[2], x[0]:x[1], y[0]:y[1]
        ],
        axis=0
    )

    sys.stdout.write('Made Image\n')

    calib_velocity = 333390.00079943583

    if calib_velocity is None:
        calib_velocity = get_calib_velocity()

    # calib_velocity = 357188.5568518038

    temp0, vlos0, vturb0 = get_atmos_values_for_lables(tau_indices_list[0])
    temp1, vlos1, vturb1 = get_atmos_values_for_lables(tau_indices_list[1])
    temp2, vlos2, vturb2 = get_atmos_values_for_lables(tau_indices_list[2])

    sys.stdout.write('Calculated Atmos Params\n')

    vlos0 -= calib_velocity
    vlos1 -= calib_velocity
    vlos2 -= calib_velocity

    vlos0 /= 1e5
    vlos1 /= 1e5
    vlos2 /= 1e5

    vturb0 /= 1e5
    vturb1 /= 1e5
    vturb2 /= 1e5

    f = h5py.File(label_file_path, 'r')

    labels = f['new_final_labels'][frames[0]:frames[1], x[0]:x[1], y[0]:y[1]].astype(np.int64)

    temp_map00 = get_image_map(labels[0], temp0)
    temp_map01 = get_image_map(labels[0], temp1)
    temp_map02 = get_image_map(labels[0], temp2)

    vlos_map00 = get_image_map(labels[0], vlos0)
    vlos_map01 = get_image_map(labels[0], vlos1)
    vlos_map02 = get_image_map(labels[0], vlos2)

    vturb_map00 = get_image_map(labels[0], vturb0)
    vturb_map01 = get_image_map(labels[0], vturb1)
    vturb_map02 = get_image_map(labels[0], vturb2)

    fig, axs = plt.subplots(4, 3, figsize=(18, 12), dpi=100)

    im00 = axs[0][0].imshow(
        image00,
        origin='lower',
        cmap='gray',
        interpolation='none'
    )
    im01 = axs[0][1].imshow(
        image01,
        origin='lower',
        cmap='gray',
        interpolation='none'
    )
    im02 = axs[0][2].imshow(
        image02,
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    im10 = axs[1][0].imshow(
        temp_map00,
        origin='lower',
        cmap='hot',
        interpolation='none'
    )
    im11 = axs[1][1].imshow(
        temp_map01,
        origin='lower',
        cmap='hot',
        interpolation='none'
    )
    im12 = axs[1][2].imshow(
        temp_map02,
        origin='lower',
        cmap='hot',
        interpolation='none'
    )

    im20 = axs[2][0].imshow(
        vlos_map00,
        origin='lower',
        cmap='bwr',
        interpolation='none',
        vmin=-2,
        vmax=2,
        aspect='equal'
    )
    im21 = axs[2][1].imshow(
        vlos_map01,
        origin='lower',
        cmap='bwr',
        interpolation='none',
        vmin=-6,
        vmax=6,
        aspect='equal'
    )
    im22 = axs[2][2].imshow(
        vlos_map02,
        origin='lower',
        cmap='bwr',
        interpolation='none',
        vmin=-6,
        vmax=6,
        aspect='equal'
    )

    im30 = axs[3][0].imshow(
        vturb_map00,
        origin='lower',
        cmap='copper',
        interpolation='none',
        vmin=0,
        vmax=5,
        aspect='equal'
    )
    im31 = axs[3][1].imshow(
        vturb_map01,
        origin='lower',
        cmap='copper',
        interpolation='none',
        vmin=0,
        vmax=5,
        aspect='equal'
    )
    im32 = axs[3][2].imshow(
        vturb_map02,
        origin='lower',
        cmap='copper',
        interpolation='none',
        vmin=0,
        vmax=5,
        aspect='equal'
    )

    axs[0][0].set_xticklabels([])
    axs[0][0].set_yticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][2].set_yticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][0].set_yticklabels([])
    axs[1][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_xticklabels([])
    axs[1][2].set_yticklabels([])
    axs[2][0].set_xticklabels([])
    axs[2][0].set_yticklabels([])
    axs[2][1].set_xticklabels([])
    axs[2][1].set_yticklabels([])
    axs[2][2].set_xticklabels([])
    axs[2][2].set_yticklabels([])
    axs[3][0].set_xticklabels([])
    axs[3][0].set_yticklabels([])
    axs[3][1].set_xticklabels([])
    axs[3][1].set_yticklabels([])
    axs[3][2].set_xticklabels([])
    axs[3][2].set_yticklabels([])

    cbar00 = fig.colorbar(im00, ax=axs[0][0])
    cbar01 = fig.colorbar(im01, ax=axs[0][1])
    cbar02 = fig.colorbar(im02, ax=axs[0][2])
    cbar10 = fig.colorbar(im10, ax=axs[1][0])
    cbar11 = fig.colorbar(im11, ax=axs[1][1])
    cbar12 = fig.colorbar(im12, ax=axs[1][2])
    cbar20 = fig.colorbar(im20, ax=axs[2][0])
    cbar21 = fig.colorbar(im21, ax=axs[2][1])
    cbar22 = fig.colorbar(im22, ax=axs[2][2])
    cbar30 = fig.colorbar(im30, ax=axs[3][0])
    cbar31 = fig.colorbar(im31, ax=axs[3][1])
    cbar32 = fig.colorbar(im32, ax=axs[3][2])

    cbar00.ax.tick_params(labelsize=10)
    cbar01.ax.tick_params(labelsize=10)
    cbar02.ax.tick_params(labelsize=10)
    cbar10.ax.tick_params(labelsize=10)
    cbar11.ax.tick_params(labelsize=10)
    cbar12.ax.tick_params(labelsize=10)
    cbar20.ax.tick_params(labelsize=10)
    cbar21.ax.tick_params(labelsize=10)
    cbar22.ax.tick_params(labelsize=10)
    cbar30.ax.tick_params(labelsize=10)
    cbar31.ax.tick_params(labelsize=10)
    cbar32.ax.tick_params(labelsize=10)

    fig.tight_layout()
    plt.tight_layout()

    def updatefig(j):
        # set the data in the axesimage object
        im00.set_array(
            np.mean(
                data[j, 0, wave_indices_list[0], x[0]:x[1], y[0]:y[1]],
                axis=0
            )
        )
        im01.set_array(
            np.mean(
                data[j, 0, wave_indices_list[1], x[0]:x[1], y[0]:y[1]],
                axis=0
            )
        )
        im02.set_array(
            np.mean(
                data[j, 0, wave_indices_list[2], x[0]:x[1], y[0]:y[1]],
                axis=0
            )
        )

        im10.set_array(
            get_image_map(labels[j], temp0)
        )
        im11.set_array(
            get_image_map(labels[j], temp1)
        )
        im12.set_array(
            get_image_map(labels[j], temp2)
        )

        im20.set_array(
            get_image_map(labels[j], vlos0)
        )
        im21.set_array(
            get_image_map(labels[j], vlos1)
        )
        im22.set_array(
            get_image_map(labels[j], vlos2)
        )

        im30.set_array(
            get_image_map(labels[j], vturb0)
        )
        im31.set_array(
            get_image_map(labels[j], vturb1)
        )
        im32.set_array(
            get_image_map(labels[j], vturb2)
        )

        fig.suptitle('Frame {}'.format(j))
        # return the artists set
        return [im00, im01, im02, im10, im11, im12, im20, im21, im22, im30, im31, im32]

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
        animation_path='rp_map_fov.mp4',
        wave_indices_list=[photosphere_indices, mid_chromosphere_indices, upper_chromosphere_indices],
        tau_indices_list=[photosphere_tau, mid_chromosphere_tau, upper_chromosphere_tau]
    )

    # plot_fov_parameter_variation(
    #     animation_path='mid_chromosphere_map.mp4',
    #     wave_indices=mid_chromosphere_indices,
    #     tau_indices=mid_chromosphere_tau
    # )

    # plot_fov_parameter_variation(
    #     animation_path='upper_chromosphere_map.mp4',
    #     wave_indices=upper_chromosphere_indices,
    #     tau_indices=upper_chromosphere_tau
    # )
