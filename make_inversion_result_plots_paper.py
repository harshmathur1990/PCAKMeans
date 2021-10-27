import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator


inversion_out_file = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/FoVAtoJ.nc')

fovNameList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

strong_shocks_profiles = np.array(
    [
        85, 36, 18, 78
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

write_path = Path(
    '/home/harsh/Shocks Paper/InversionMaps/'
)

def make_line_cut_plots(all_params, time_array, mask, fovName):

    all_vmin = np.zeros(3)
    all_vmax = np.zeros(3)

    all_vmin[0] = np.round(
        all_params[:, 0].min(),
        2
    )
    all_vmax[0] = np.round(
        all_params[:, 0].max(),
        2
    )

    all_vmin[1] = -8
    all_vmax[1] = 8

    all_vmin[2] = 0
    all_vmax[2] = 6

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(2, 3)

    gs.update(wspace=0.0, hspace=0.0)

    X, Y = np.meshgrid(range(50), ltau)

    k = 0

    for i in range(2):
        for j in range(3):
            axs = fig.add_subplot(gs[k])

            if j == 0:
                cmap = 'hot'

            elif j == 1:
                cmap='bwr'
            else:
                cmap='copper'

            im = axs.pcolormesh(
                X, Y,
                all_params[i, j],
                cmap=cmap,
                vmin=all_vmin[j],
                vmax=all_vmax[j],
                shading='gouraud'
            )

            axs.tick_params(labelsize=9, colors='black')

            axs.invert_yaxis()

            axs.contour(
                X, Y,
                mask[i],
                levels=0,
                cmap='gray'
            )

            axs.set_xticks([])
            axs.set_yticks([])

            axs.set_xticklabels([])
            axs.set_yticklabels([])

            if i == 0:

                cbaxes = inset_axes(
                    axs,
                    width="80%",
                    height="3%",
                    loc='upper center',
                    borderpad=-1
                )
                cbar = fig.colorbar(
                    im,
                    cax=cbaxes,
                    ticks=[all_vmin[j], all_vmax[j]],
                    orientation='horizontal'
                )
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.tick_params(labelsize=9, colors='black')

            if j == 0:

                axs.text(
                    0.05, 0.05, r't={}s'.format(
                        time_array[i]
                    ),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='small'
                )

                axs.set_yticks([-7, -6, -5, -4, -3, -2, -1, 0])
                axs.set_yticklabels([-7, -6, -5, -4, -3, -2, -1, 0])

                if i == 0:
                    axs.text(
                        0.05, 0.8, r'FoV {}'.format(
                            fovName
                        ),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='small'
                    )
            k += 1

            if i == 1:
                axs.set_xticks([10, 20, 30, 40])
                axs.set_xticklabels([10, 20, 30, 40])

    fig.savefig(
        write_path / 'FoV_{}_linecut.pdf'.format(
            fovName
        ),
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def plot_data_for_result_plots(index, start_t, mark_t, mark_y):

    time = np.round(np.arange(0, 826, 8.26), 2)

    f = h5py.File(inversion_out_file, 'r')

    labels_mask = np.zeros((2, 150, 50), dtype=np.int64)

    for ltau_val in [-4.2, -3, -1]:

        ltau_index = np.argmin(np.abs(ltau - ltau_val))

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(7, 3))

        gs = gridspec.GridSpec(3, 7)

        gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

        temp = f['all_temp'][index * 7:index * 7 + 7, :, :, ltau_index] / 1e3
        vlos = f['all_vlos'][index * 7:index * 7 + 7, :, :, ltau_index]
        vturb = f['all_vturb'][index * 7:index * 7 + 7, :, :, ltau_index]
        labels = f['all_labels'][index * 7:index * 7 + 7]

        # temp_vmin = [np.round(temp[i].min(), 2) for i in range(7)]
        # temp_vmax = [np.round(temp[i].max(), 2) for i in range(7)]

        if ltau_val == -1:
            temp_vmin = np.round(temp.min(), 2)
            temp_vmax = np.round(temp.max(), 2)
        elif ltau_val == -3:
            temp_vmin = np.round(temp.min(), 2)
            temp_vmax = np.round(temp.max(), 2)
        else:
            temp_mn = np.round(temp.mean(), 2)
            temp_sd = np.round(temp.std(), 2)
            temp_vmin = np.round(temp.min(), 2)
            temp_vmax = temp_mn + 4 * temp_sd

        if ltau_val == -4.2:
            vlos_vmin = -8
            vlos_vmax = 8
        elif ltau_val == -3:
            vlos_vmin = -6
            vlos_vmax = 6
        else:
            vlos_vmin = -3
            vlos_vmax = 3

        vturb_vmin = 0
        vturb_vmax = 5

        if ltau_val == -4.2:
            color = 'white'
        else:
            color = 'black'
        k = 0
        for i in range(3):
            for j in range(7):
                axs = fig.add_subplot(gs[k])
                if i == 0:
                    im = axs.imshow(temp[j], cmap='hot', origin='lower', vmin=temp_vmin, vmax=temp_vmax)
                    if j == 0:
                        cbaxes = inset_axes(
                            axs,
                            width="40%",
                            height="5%",
                            loc=4,
                            borderpad=1
                        )
                        cbar = fig.colorbar(
                            im,
                            cax=cbaxes,
                            ticks=[temp_vmin, temp_vmax],
                            orientation='horizontal'
                        )

                        cbar.ax.xaxis.set_ticks_position('top')
                        cbar.ax.tick_params(labelsize=8, colors=color)
                    if j == 0:
                        axs.text(
                            0.4, 0.9,
                            'FoV {}'.format(
                                fovNameList[index]
                            ),
                            transform=axs.transAxes,
                            color=color,
                            fontsize='small'
                        )
                    axs.text(
                        0.4, 0.7,
                        't={}s'.format(
                            time[start_t + j]
                        ),
                        transform=axs.transAxes,
                        color=color,
                        fontsize='small'
                    )
                    if j == 6:
                        axs.text(
                            0.1, 0.5,
                            r'$\log(\tau_{{500}})={}$'.format(
                                ltau_val
                            ),
                            transform=axs.transAxes,
                            color=color,
                            fontsize='x-small'
                        )
                elif i == 1:
                    im = axs.imshow(vlos[j], cmap='bwr', origin='lower', vmin=vlos_vmin, vmax=vlos_vmax)
                    if j == 0:
                        cbaxes = inset_axes(
                            axs,
                            width="40%",
                            height="5%",
                            loc=4,
                            borderpad=1
                        )
                        cbar = fig.colorbar(
                            im,
                            cax=cbaxes,
                            ticks=[vlos_vmin, vlos_vmax],
                            orientation='horizontal'
                        )

                        cbar.ax.xaxis.set_ticks_position('top')
                        clr = color
                        if ltau_val == -4.2:
                            clr = 'black'
                        cbar.ax.tick_params(labelsize=9, colors=clr)
                else:
                    im = axs.imshow(vturb[j], cmap='copper', origin='lower', vmin=vturb_vmin, vmax=vturb_vmax)
                    if j == 0:
                        cbaxes = inset_axes(
                            axs,
                            width="40%",
                            height="5%",
                            loc=4,
                            borderpad=1
                        )
                        cbar = fig.colorbar(
                            im,
                            cax=cbaxes,
                            ticks=[vturb_vmin, vturb_vmax],
                            orientation='horizontal'
                        )

                        cbar.ax.xaxis.set_ticks_position('top')
                        cbar.ax.tick_params(labelsize=9, colors='white')

                mask = np.zeros((50, 50), dtype=np.int64)

                for profile in list(strong_shocks_profiles):
                    mask[np.where(labels[j] == profile)] = 1

                if j == 0 or j == (mark_t - start_t):
                    axs.axvline(x=mark_y, linestyle='--', color='blue')
                    if i == 0:
                        labels_mask[0] = mask[:, mark_y][np.newaxis, :]
                    else:
                        labels_mask[1] = mask[:, mark_y][np.newaxis, :]

                X, Y = np.meshgrid(range(50), range(50))
                axs.contour(X, Y, mask, levels=1, color='blue')

                axs.set_xticks([])
                axs.set_yticks([])
                axs.set_xticklabels([])
                axs.set_yticklabels([])
                k += 1

        fig.savefig(
            write_path / 'Inversion_FoV_{}_lt_{}.pdf'.format(
                fovNameList[index],
                ltau_val
            ),
            dpi=300,
            format='pdf'
        )

    plt.close('all')

    plt.clf()

    plt.cla()

    fovName = fovNameList[index]

    time_array = time[np.array([start_t, mark_t])]

    all_params = np.zeros((2, 3, 150, 50), dtype=np.float64)

    indice = np.array([index * 7, index * 7 + mark_t -start_t])

    all_params[:, 0] = np.transpose(
        f['all_temp'][indice, :, mark_y] / 1e3,
        axes=(0, 2, 1)
    )

    all_params[:, 1] = np.transpose(
        f['all_vlos'][indice, :, mark_y],
        axes=(0, 2, 1)
    )

    all_params[:, 2] = np.transpose(
        f['all_vturb'][indice, :, mark_y],
        axes=(0, 2, 1)
    )

    make_line_cut_plots(all_params, time_array, labels_mask, fovName)


def make_time_evolution_plots(index_f, start_t, mark_x, mark_y):
    vmin = -6
    vmax = 6
    tmin = -2
    tmax = 3.5

    log_t_values = np.array([-4.2, -3, -1])

    params = np.zeros((2, log_t_values.size, 7))

    ind_lt = list()

    for log_t in log_t_values:
        ind = np.argmin(np.abs(ltau - log_t))
        ind_lt.append(ind)

    ind_lt = np.array(ind_lt)

    f = h5py.File(inversion_out_file, 'r')

    params[0] = np.transpose(
        f['all_temp'][index_f * 7:index_f * 7 + 7, mark_x, mark_y, ind_lt],
        axes=(1, 0)
    ) / 1e3
    params[1] = np.transpose(
        f['all_vlos'][index_f * 7:index_f * 7 + 7, mark_x, mark_y, ind_lt],
        axes=(1, 0)
    )

    # params[0] -= np.mean(params[0], 1)[:, np.newaxis]

    params[0] = params[0] - params[0, :, 0][:, np.newaxis]

    mask_shock = np.zeros(7, dtype=np.int64)

    labels_f = f['all_labels'][index_f * 7:index_f * 7 + 7, mark_x, mark_y]

    for profile in strong_shocks_profiles:
        mask_shock[np.where(labels_f == profile)] = 1

    time = np.round(np.arange(0, 826, 8.26), 2)

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    k = 0

    gs = gridspec.GridSpec(2, 1)

    for i in range(2):
        axs = fig.add_subplot(gs[k])

        for index, (log_t, color) in enumerate(
                zip(
                    log_t_values,
                    ['blue', 'green', 'orange']
                )
        ):

            if k == 0:
                axs.plot(
                    time[start_t:start_t+7],
                    params[k, index],
                    color=color,
                    label=r'$\log (\tau_{{500}})={}$'.format(
                        log_t
                    )
                )

                handles, labels = axs.get_legend_handles_labels()

                plt.legend(
                    handles,
                    labels,
                    ncol=3,
                    bbox_to_anchor=(0., 1.02, 1., .102),
                    loc='lower left',
                    mode="expand",
                    borderaxespad=0.
                )

                axs.set_ylim(tmin, tmax)
                axs.set_yticks(np.arange(tmin + 0.5, tmax, 0.5))
                axs.set_yticklabels(np.arange(tmin + 0.5, tmax, 0.5))
                axs.set_ylabel(r'$\delta T[kK]$')
                axs.yaxis.set_minor_locator(MultipleLocator(0.1))
            else:
                axs.plot(
                    time[start_t:start_t+7],
                    params[k, index],
                    color=color
                )

                axs.set_ylim(vmin, vmax)
                axs.set_yticks(np.arange(vmin + 2, vmax, 2))
                axs.set_yticklabels(np.arange(vmin + 2, vmax, 2))
                axs.set_ylabel(r'$V_{LOS}[kms^{-1}]$')
                axs.set_xlabel(r'$Time\;(s)$')
                axs.yaxis.set_minor_locator(MultipleLocator(0.5))

            axs.grid(True, ls='--', alpha=0.5)
            axs.xaxis.set_minor_locator(MultipleLocator(10))
            axs.tick_params(direction='in', which='both')

        ind_shocks = np.where(mask_shock == 1)[0]

        for ind in ind_shocks:
            axs.axvline(
                time[start_t + ind],
                linestyle='--',
                linewidth=0.5,
                color='black'
            )

        axs.text(
            0.1, 0.9,
            'FoV {}'.format(fovNameList[index_f]),
            transform=axs.transAxes,
            color='black',
            fontsize='small'
        )
        k += 1

    fig.tight_layout()

    fig.savefig(
        write_path / 'FoV_{}_param_variation.pdf'.format(fovNameList[index_f]),
        dpi=300,
        format='pdf'
    )

if __name__ == '__main__':
    plot_data_for_result_plots(0,  4,  6, 18)
    plot_data_for_result_plots(1, 14, 16, 15)
    plot_data_for_result_plots(2, 17, 20, 27)
    plot_data_for_result_plots(3, 32, 35, 20)
    plot_data_for_result_plots(4, 12, 15, 22)
    plot_data_for_result_plots(5, 57, 59, 28)
    plot_data_for_result_plots(6, 93, 97, 22)
    plot_data_for_result_plots(7,  7, 11, 16)
    plot_data_for_result_plots(8,  8, 11, 21)
    plot_data_for_result_plots(9,  9, 12, 28)
    make_time_evolution_plots(0,  4, 25, 18)
    make_time_evolution_plots(1, 14, 16, 15)
    make_time_evolution_plots(2, 17, 23, 27)
    make_time_evolution_plots(3, 32, 29, 20)
    make_time_evolution_plots(4, 12, 24, 22)
    make_time_evolution_plots(5, 57, 28, 28)
    make_time_evolution_plots(6, 93, 24, 22)
    make_time_evolution_plots(7,  7, 22, 16)
    make_time_evolution_plots(8,  8, 26, 21)
    make_time_evolution_plots(9,  9, 28, 28)