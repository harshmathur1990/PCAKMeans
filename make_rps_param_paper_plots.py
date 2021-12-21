import sys
import numpy as np
import h5py
import sunpy.io
import matplotlib.pyplot as plt
from helita.io.lp import *
import matplotlib.gridspec as gridspec
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from calculate_calib_velocity_and_classify_rps import get_shocks_mask, \
    get_very_strong_shocks_mask, get_very_very_strong_shocks_mask

base_path = Path('/home/harsh/OsloAnalysis')

input_file_3950 = base_path / 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
old_kmeans_file = base_path / 'new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

new_kmeans = base_path / 'new_kmeans'

all_data_inversion_rps = new_kmeans / 'all_data_inversion_rps'

strong_shocks_profiles = np.array(
    [
        85, 36, 18, 78
    ]
)

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4.5, -3.5])

upper_chromosphere_tau = np.array([-5.5, -4.5])

cont_value = [2.4434714e-05, 4.2277254e-08, 4.054384e-08]

shocks_paper_path = Path('/home/harsh/Shocks Paper/')

ltau = np.array(
    [
        -8.       , -7.78133  , -7.77448  , -7.76712  , -7.76004  ,
        -7.75249  , -7.74429  , -7.7356   , -7.72638  , -7.71591  ,
        -7.70478  , -7.69357  , -7.68765  , -7.68175  , -7.67589  ,
        -7.66997  , -7.66374  , -7.65712  , -7.64966  , -7.64093  ,
        -7.63093  , -7.6192   , -7.6053   , -7.58877  , -7.56925  ,
        -7.54674  , -7.52177  , -7.49317  , -7.4585   , -7.41659  ,
        -7.36725  , -7.31089  , -7.24834  , -7.18072  , -7.1113   ,
        -7.04138  , -6.97007  , -6.89698  , -6.82299  , -6.74881  ,
        -6.67471  , -6.60046  , -6.52598  , -6.45188  , -6.37933  ,
        -6.30927  , -6.24281  , -6.17928  , -6.11686  , -6.05597  ,
        -5.99747  , -5.94147  , -5.88801  , -5.84684  , -5.81285  ,
        -5.78014  , -5.74854  , -5.71774  , -5.68761  , -5.65825  ,
        -5.6293   , -5.60066  , -5.57245  , -5.54457  , -5.51687  ,
        -5.48932  , -5.46182  , -5.43417  , -5.40623  , -5.37801  ,
        -5.3496   , -5.32111  , -5.29248  , -5.26358  , -5.23413  ,
        -5.20392  , -5.17283  , -5.14073  , -5.1078   , -5.07426  ,
        -5.03999  , -5.00492  , -4.96953  , -4.93406  , -4.89821  ,
        -4.86196  , -4.82534  , -4.78825  , -4.75066  , -4.71243  ,
        -4.67439  , -4.63696  , -4.59945  , -4.5607   , -4.52212  ,
        -4.48434  , -4.44653  , -4.40796  , -4.36863  , -4.32842  ,
        -4.28651  , -4.24205  , -4.19486  , -4.14491  , -4.09187  ,
        -4.03446  , -3.97196  , -3.90451  , -3.83088  , -3.7496   ,
        -3.66     , -3.56112  , -3.4519   , -3.33173  , -3.20394  ,
        -3.07448  , -2.94444  , -2.8139   , -2.68294  , -2.55164  ,
        -2.42002  , -2.28814  , -2.15605  , -2.02377  , -1.89135  ,
        -1.7588   , -1.62613  , -1.49337  , -1.36127  , -1.23139  ,
        -1.10699  , -0.99209  , -0.884893 , -0.782787 , -0.683488 ,
        -0.584996 , -0.485559 , -0.383085 , -0.273456 , -0.152177 ,
        -0.0221309,  0.110786 ,  0.244405 ,  0.378378 ,  0.51182  ,
        0.64474  ,  0.777188 ,  0.909063 ,  1.04044  ,  1.1711
    ]
)


def get_atmos_values_for_lables(ltau_val_min, ltau_val_max):

    indices = np.where((ltau >= ltau_val_min) & (ltau <= ltau_val_max))[0]

    temp = np.zeros((100, 150))

    vlos = np.zeros((100, 150))

    vturb = np.zeros((100, 150))

    for i in range(100):
        filename, content_list = get_filepath_and_content_list(i)

        f = h5py.File(filename, 'r')

        index = content_list.index(i)

        if len(content_list) == f['ltau500'].shape[1]:
            normal = True
        else:
            normal = False

        if normal:

            temp[i] = f['temp'][0, index, 0]

            vlos[i] = f['vlos'][0, index, 0]

            vturb[i] = f['vturb'][0, index, 0]

        else:

            temp[i] = f['temp'][0, 0, index]

            vlos[i] = f['vlos'][0, 0, index]

            vturb[i] = f['vturb'][0, 0, index]

        f.close()

    temp = np.mean(temp[:, indices], 1)

    vlos = np.mean(vlos[:, indices], 1)

    vturb = np.mean(vturb[:, indices], 1)

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


def prepare_get_parameter(param):

    def get_parameter(rp):
        return param[rp]

    return get_parameter


def get_filepath_and_content_list(rp):
    if rp in [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]:
        filename, content_list = all_data_inversion_rps / 'quiet/plots_v1/wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63_cycle_1_t_6_vl_3_vt_4_atmos.nc', [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]

    elif rp in [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]:
        filename, content_list = all_data_inversion_rps / 'shocks/plots_v1/wholedata_rps_shock_profiles_rps_2_4_10_19_26_30_37_52_79_85_94_1_22_23_53_55_56_66_67_72_77_80_81_92_87_99_36_6_49_17_96_98_cycle_1_t_6_vl_5_vt_4_atmos.nc', [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]

    elif rp in [78, 18]:
        filename, content_list = all_data_inversion_rps / 'shocks_78_18/plots_v1/wholedata_rps_shocks_78_18_profile_rps_78_18_cycle_1_t_5_vl_5_vt_4_atmos.nc', [78, 18]

    elif rp in [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]:
        filename, content_list = all_data_inversion_rps / 'reverse/plots_v1/wholedata_rps_reverse_shock_profiles_rps_3_13_16_25_32_33_35_41_45_46_58_61_64_68_82_95_97_cycle_1_t_6_vl_5_vt_4_atmos.nc', [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]

    elif rp in [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]:
        filename, content_list = all_data_inversion_rps / 'other/plots_v1/wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93_cycle_1_t_6_vl_5_vt_4_atmos.nc', [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]

    return filename, content_list


def get_data_for_rps_guess_map_plot():
    f = h5py.File(old_kmeans_file, 'r')
    labels = f['new_final_labels'][6].astype(np.int64)
    f.close()

    data, header = sunpy.io.fits.read(input_file_3950)[0]

    image00 = np.mean(
        data[
            6, 0, photosphere_indices, 662:712, 708:758
        ] / cont_value[0],
        axis=0
    )
    image01 = np.mean(
        data[
            6, 0, mid_chromosphere_indices, 662:712, 708:758
        ] / cont_value[0],
        axis=0
    )
    image02 = np.mean(
        data[
            6, 0, upper_chromosphere_indices, 662:712, 708:758
        ] / cont_value[0],
        axis=0
    )

    temp0, vlos0, vturb0 = get_atmos_values_for_lables(*photosphere_tau)
    temp1, vlos1, vturb1 = get_atmos_values_for_lables(*mid_chromosphere_tau)
    temp2, vlos2, vturb2 = get_atmos_values_for_lables(*upper_chromosphere_tau)

    temp0 /= 1e3
    temp1 /= 1e3
    temp2 /= 1e3

    vlos0 /= 1e5
    vlos1 /= 1e5
    vlos2 /= 1e5

    vturb0 /= 1e5
    vturb1 /= 1e5
    vturb2 /= 1e5

    temp_map00 = get_image_map(labels[662:712, 708:758], temp0)
    temp_map01 = get_image_map(labels[662:712, 708:758], temp1)
    temp_map02 = get_image_map(labels[662:712, 708:758], temp2)

    vlos_map00 = get_image_map(labels[662:712, 708:758], vlos0)
    vlos_map01 = get_image_map(labels[662:712, 708:758], vlos1)
    vlos_map02 = get_image_map(labels[662:712, 708:758], vlos2)

    vturb_map00 = get_image_map(labels[662:712, 708:758], vturb0)
    vturb_map01 = get_image_map(labels[662:712, 708:758], vturb1)
    vturb_map02 = get_image_map(labels[662:712, 708:758], vturb2)

    params = np.zeros((4, 3, 50, 50), dtype=np.float64)

    params[0, 0] = image00
    params[0, 1] = image01
    params[0, 2] = image02

    params[1, 0] = temp_map00
    params[1, 1] = temp_map01
    params[1, 2] = temp_map02

    params[2, 0] = vlos_map00
    params[2, 1] = vlos_map01
    params[2, 2] = vlos_map02

    params[3, 0] = vturb_map00
    params[3, 1] = vturb_map01
    params[3, 2] = vturb_map02

    return params, labels[662:712, 708:758]


def plot_paper_rp_guess_map_plot():

    write_path = Path('/home/harsh/Shocks Paper/KMeans/')

    params, labels = get_data_for_rps_guess_map_plot()

    X, Y = np.meshgrid(range(50), range(50))

    fontsize = 8

    lightblue = '#5089C6'
    mediumdarkblue = '#035397'
    darkblue = '#001E6C'
    lwidth = 1

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(3.5, 7))

    # [0][0]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0, right=0.333, bottom=0.732, top=0.932, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    this_axs.imshow(params[0][0], cmap='gray', origin='lower')
    this_axs.set_ylim(0, 60)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.05, 0.9, r'Continuum 4000 $\mathrm{\AA}$', transform=this_axs.transAxes, fontsize=fontsize)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    # [0][1]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.333, right=0.666, bottom=0.732, top=0.932, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    this_axs.imshow(params[0][1], cmap='gray', origin='lower')
    this_axs.set_ylim(0, 60)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.1, 0.9, r'Ca II K wing', transform=this_axs.transAxes, fontsize=fontsize)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    # [0][2]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.666, right=1.0, bottom=0.732, top=0.932, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    this_axs.imshow(params[0][2], cmap='gray', origin='lower')
    this_axs.set_ylim(0, 60)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.1, 0.9, r'Ca II K inner core', transform=this_axs.transAxes, fontsize=fontsize)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    # [1][0]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0, right=0.333, bottom=0.466, top=0.732, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[1][0], cmap='hot', origin='lower', vmin=5.7, vmax=5.9)
    this_axs.set_ylim(0, 80)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.1, 0.9, r'${}<\log \tau_{{\mathrm{{500}}}}<{}$'.format(-1, 0), transform=this_axs.transAxes, fontsize=7)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=2.5
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[5.7, 5.9],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [1][1]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.333, right=0.666, bottom=0.466, top=0.732, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[1][1], cmap='hot', origin='lower', vmin=4.2, vmax=5.8)
    this_axs.set_ylim(0, 80)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.01, 0.9, r'${}<\log \tau_{{\mathrm{{500}}}}<{}$'.format(-4.5, -3.5), transform=this_axs.transAxes,
                   fontsize=7)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=2.5
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[4.2, 5.8],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [1][2]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.666, right=1.0, bottom=0.466, top=0.732, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[1][2], cmap='hot', origin='lower', vmin=4.7, vmax=6.9)
    this_axs.set_ylim(0, 80)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    this_axs.text(0.01, 0.9, r'${}<\log \tau_{{\mathrm{{500}}}}<{}$'.format(-5.5, -4.5), transform=this_axs.transAxes,
                   fontsize=7)
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=2.5
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[4.7, 6.9],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [2][0]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0, right=0.333, bottom=0.233, top=0.466, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[2][0], cmap='bwr', origin='lower', vmin=-2, vmax=2)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[-2, 2],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [2][1]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.333, right=0.666, bottom=0.233, top=0.466, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[2][1], cmap='bwr', origin='lower', vmin=-2, vmax=2)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[-2, 2],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [2][2]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.666, right=1.0, bottom=0.233, top=0.466, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[2][2], cmap='bwr', origin='lower', vmin=-6, vmax=6)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[-6, 6],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [3][0]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0, right=0.333, bottom=0.0, top=0.233, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[3][0], cmap='copper', origin='lower', vmin=0, vmax=4)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[0, 4],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [3][1]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.333, right=0.666, bottom=0.0, top=0.233, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[3][1], cmap='copper', origin='lower', vmin=0, vmax=4)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[0, 4],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    # [3][2]
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.666, right=1.0, bottom=0.0, top=0.233, wspace=0.0, hspace=0.0)
    this_axs = fig.add_subplot(gs[0])
    im = this_axs.imshow(params[3][2], cmap='copper', origin='lower', vmin=0, vmax=4)
    this_axs.set_ylim(0, 70)
    this_axs.set_xticks([])
    this_axs.set_yticks([])
    this_axs.set_xticklabels([])
    this_axs.set_yticklabels([])
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    this_axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=lwidth,
        alpha=1,
        levels=0
    )

    cbaxes = inset_axes(
        this_axs,
        width="70%",
        height="7%",
        loc='upper center',
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[0, 4],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=fontsize, colors='black')

    fig.suptitle('FoV A', fontsize=fontsize)

    fig.savefig(write_path / 'InversionGuessRPs.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


if __name__ == '__main__':
    plot_paper_rp_guess_map_plot()
