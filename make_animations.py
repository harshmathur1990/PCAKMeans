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
