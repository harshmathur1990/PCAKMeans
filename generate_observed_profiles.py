import sunpy.io
import numpy as np
import h5py
from prepare_data import *


f = h5py.File('out_45.h5', 'r')
selected_frames = f['selected_frames'][()]
labels = f['labels_'][()].reshape(7, 1236, 1848)
data, header = sunpy.io.fits.read('/Volumes/Harsh 9599771751/colabd/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')[0]
seldata = data[selected_frames, 0]
representative_profiles = list()

for k in range(45):
    a, b, c = np.where(labels == k)

    representative_profiles.append(
        np.median(seldata[a, :, b, c], axis=0)
    )

wck, ick = findgrid(wave[0:29], (wave[2] - wave[1]), extra=8)

ca_k = sp.profile(nx=45, ny=1, ns=4, nw=wck.size + 1)

for i in range(45):
    cw = np.asarray([4000.])
    cont = []
    for ii in cw:
        cont.append(getCont(ii))

    ca_k.wav[0:-1] = wck[:]

    ca_k.wav[-1] = wave[-1]

    ca_k.dat[0, 0, i, ick, 0] = representative_profiles[i][:29] / cont[0]
    ca_k.dat[0, 0, i, -1, 0] = representative_profiles[i][-1] / cont[0]

ca_k.weights[:, :] = 1.e16
ca_k.weights[ick, 0] = 0.002
ca_k.weights[-1, 0] = 0.004

sp_all = ca_k

sp_all.write('merged_rps.nc'.format(i))

lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"

print(
    lab.format(
        ca_k.wav[0],
        ca_k.wav[1] - ca_k.wav[0],
        ca_k.wav.size - 1, cont[0],
        'fpi, 3934.nc'
    )
)
print(
    lab.format(
        ca_k.wav[-1],
        ca_k.wav[1] - ca_k.wav[0],
        1,
        cont[0],
        'none, none'
    )
)

    # dw = ca_k.wav[1] - ca_k.wav[0]
    # ntw = 25
    # tw1 = (np.arange(ntw) - ntw // 2) * dw + 3934.0
    # tr1 = cr.dual_fpi(tw1)
    # tr1 /= tr1.sum()
    # # Stores the FPI profile and the parameters of the prefilter
    # writeInstProf('3934.nc', tr1, [ca_k.wav[ick[29 // 2]], 4.5, 3.0])
