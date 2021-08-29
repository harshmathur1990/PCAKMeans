import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


wave_3933 = np.array(
    [
        3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
        3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
        3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
        3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
        3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
        3934.42352, 3934.48888, 3934.55424, 3934.6196, 4001.14744
    ]
)

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


def get_doppler_velocity(wavelength, center_wavelength):
    return (wavelength - center_wavelength) * 2.99792458e5 / center_wavelength


@np.vectorize
def get_doppler_velocity_3950(wavelength):
    return get_doppler_velocity(wavelength, 3933.682)


@np.vectorize
def get_relative_velocity(wavelength):
    return wavelength - 3933.682


def plot_fov_results_for_a_pixel(
    x, y, ref_x, ref_y, t, fovName
):

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        2
    )

    size = plt.rcParams['lines.markersize']

    rela_wave = get_relative_velocity(wave_3933[:-1])

    f = h5py.File(out_file, 'r')

    all_profiles = f['all_profiles']
    syn_profiles = f['syn_profiles']
    all_temp = f['all_temp']
    all_vlos = f['all_vlos']
    all_vturb = f['all_vturb']

    #------------------------------#

    plt.cla()

    plt.clf()

    plt.close('all')

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(2, 2)

    # gs.update(wspace=0.75, hspace=0.75)

    #------------------------------#
    axs = fig.add_subplot(gs[0])

    axs.scatter(
        rela_wave,
        all_profiles[t, ref_x, ref_y, 0:29],
        s=size/4,
        color='#0A1931'
    )

    axs.plot(
        rela_wave,
        syn_profiles[t, ref_x, ref_y, 0:29],
        '--',
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$\lambda\;(\AA)$')
    axs.set_ylabel(r'$I/I_{c 4000 \AA}$')

    axs.set_xticks([-0.5, 0, 0.5])
    axs.set_xticklabels([-0.5, 0, 0.5])

    axs.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    axs.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])

    axs.set_ylim(0, 0.4)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            time[t], fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[1])

    axs.plot(
        ltau,
        all_temp[t, ref_x, ref_y] / 1e3,
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$T[kK]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([3, 5, 7, 9, 11])
    axs.set_yticklabels([3, 5, 7, 9, 11])

    axs.set_ylim(3, 11)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            time[t], fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[2])

    axs.plot(
        ltau,
        all_vlos[t, ref_x, ref_y],
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{LOS}[kms^{-1}]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([-6, -3, 0, 3, 6])
    axs.set_yticklabels([-6, -3, 0, 3, 6])

    axs.set_ylim(-8, 8)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            time[t], fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[3])

    axs.plot(
        ltau,
        all_vturb[t, ref_x, ref_y],
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{turb}[kms^{-1}]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([0, 2, 4, 6])
    axs.set_yticklabels([0, 2, 4, 6])

    axs.set_ylim(0, 6)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            time[t], fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    fig.tight_layout()

    fig.savefig(
        'fov_{}_{}_{}_{}_t_{}_x_{}_y_{}.pdf'.format(
            x[0], x[1], y[0], y[1], t, ref_x, ref_y
        ),
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close('all')

    plt.cla()

    plt.clf()

    f.close()


if __name__ == '__main__':

    x = [662, 712]
    y = [708, 758]

