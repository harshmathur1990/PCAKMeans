import numpy as np
import matplotlib.pyplot as plt


wave_list = [
    np.array(
        [
            3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
            3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
            3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
            3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
            3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
            3934.42352, 3934.48888, 3934.55424, 3934.6196
        ]
    ),
    np.array(
        [
            8540.3941552, 8540.9941552, 8541.2341552, 8541.3941552,
            8541.5541552, 8541.7141552, 8541.8341552, 8541.9141552,
            8541.9941552, 8542.0741552, 8542.1541552, 8542.2341552,
            8542.3141552, 8542.4341552, 8542.5941552, 8542.7541552,
            8542.9141552, 8543.1541552, 8543.7541552, 8544.4541552
        ]
    ),
    np.array(
        [
            6172.9802566, 6173.0602566, 6173.1402566, 6173.1802566,
            6173.2202566, 6173.2602566, 6173.3002566, 6173.3402566,
            6173.3802566, 6173.4202566, 6173.4602566, 6173.5402566,
            6173.6202566, 6173.9802566
        ]
    )
]


wei = np.ones((64)) * 1e16
wei[0:30] = 0.001
wei[30:50] = 0.004
# wei[np.arange(30,50)[4:16]] = 0.008
wei[50:64] = 0.002
# wei[np.arange(50,64)[4:10]] = 0.004


def plot():

    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8, forward=True)

    plt.plot(wave_list[1], wei[30:50], label='Ca II 8542', color='#810000')
    plt.gca().set_ylim(0, 0.005)
    plt.gca().legend(loc='upper left')
    plt.gca().tick_params(axis='x', colors="#810000")
    plt.gca().tick_params(axis='y', colors="#000000")

    axs_0_alt = fig.add_subplot(111, label='alt_1', frame_on=False)
    axs_0_alt.plot(wave_list[0], wei[0:29], label='Ca II k', color='#185ADB')
    axs_0_alt.xaxis.tick_top()
    axs_0_alt.yaxis.tick_right()
    axs_0_alt.set_yticklabels([])
    axs_0_alt.set_xticklabels([])
    axs_0_alt.set_ylim(0, 0.005)
    axs_0_alt.legend(loc='upper center')
    axs_0_alt.tick_params(axis='x', colors="#185ADB")
    axs_0_alt.tick_params(axis='y', colors="#185ADB")

    axs_1_alt = fig.add_subplot(111, label='alt_2', frame_on=False)
    axs_1_alt.plot(wave_list[2], wei[50:64], label='Fe I 6173', color='#FC5404')
    axs_1_alt.xaxis.tick_top()
    axs_1_alt.yaxis.tick_right()
    axs_1_alt.set_yticklabels([])
    axs_1_alt.set_ylim(0, 0.005)
    axs_1_alt.legend(loc='upper right')
    axs_1_alt.tick_params(axis='x', colors="#FC5404")
    axs_1_alt.tick_params(axis='y', colors="#FC5404")

    plt.savefig('Weights.png', dpi=300, format='png')
