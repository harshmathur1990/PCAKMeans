import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sunpy.io.fits


def make_wideband_animation(
    filename,
    animation_path,
    fps=6,
    x1=None,
    x2=None,
    y1=None,
    y2=None
):

    rate = 1000 / fps

    fig = plt.figure()  # make figure

    data, header = sunpy.io.fits.read(filename)[0]

    imagelist = [
        data[i][0][0][x1:x2, y1:y2] for i in np.arange(100)
    ]

    im = plt.imshow(
        imagelist[0],
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        plt.title('Frame {}'.format(j))
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(len(imagelist)),
        interval=rate, blit=True
    )

    Writer = animation.writers['ffmpeg']

    writer = Writer(
        fps=fps,
        metadata=dict(artist='Harsh Mathur'),
        bitrate=1800
    )

    ani.save(animation_path, writer=writer)

    plt.cla()

    plt.close(fig)

    plt.close('all')


def make_narrow_animation(
    filename,
    animation_path,
    fps=6,
    wavelength=None,
    x1=None,
    x2=None,
    y1=None,
    y2=None
):

    plt.close('all')

    plt.cla()

    plt.clf()

    rate = 1000 / fps

    fig = plt.figure()  # make figure

    data, header = sunpy.io.fits.read(filename)[0]

    imagelist = [
        data[i, 0, wavelength] for i in np.arange(100)
    ]

    im = plt.imshow(
        imagelist[0],
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        plt.title('Frame {}'.format(j))
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(len(imagelist)),
        interval=rate, blit=True
    )

    Writer = animation.writers['ffmpeg']

    writer = Writer(
        fps=fps,
        metadata=dict(artist='Harsh Mathur'),
        bitrate=1800
    )

    ani.save(animation_path, writer=writer)

    plt.cla()

    plt.close(fig)

    plt.close('all')


def make_wavelength_animation(
    filename,
    animation_path,
    x, y, t1, t2,
    fps=6
):

    si_ref = 4.2277257e-05

    plt.close('all')

    plt.clf()

    plt.cla()

    rate = 1000 / fps

    fig = plt.figure()  # make figure

    data, header = sunpy.io.fits.read(filename)[0]

    data_list = data[t1:t2, 0, :-1, x, y] / si_ref

    wave = np.array(
        [
            393.278952, 393.285488, 393.292024, 393.29856,
            393.304944, 393.31148, 393.318016, 393.324552,
            393.330936, 393.337472, 393.344008, 393.350544,
            393.356928, 393.363464, 393.37, 393.376536,
            393.383072, 393.389456, 393.395992, 393.402528,
            393.409064, 393.415448, 393.421984, 393.42852,
            393.435056, 393.44144, 393.447976, 393.454512,
            393.461048
        ]
    )

    im, = plt.plot(
        wave,
        data_list[0]
    )

    plt.ylim(0, 0.3)

    def updatefig(j):
        # set the data in the axesimage object
        im.set_ydata(data_list[j - t1])
        plt.title('Frame {}'.format(j))
        # return the artists set
        return [im]

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=np.arange(t1, t2, dtype=np.int64),
        interval=rate, blit=True
    )

    Writer = animation.writers['ffmpeg']

    writer = Writer(
        fps=fps,
        metadata=dict(artist='Harsh Mathur'),
        bitrate=1800
    )

    ani.save(animation_path, writer=writer)

    plt.cla()

    plt.close(fig)

    plt.close('all')
