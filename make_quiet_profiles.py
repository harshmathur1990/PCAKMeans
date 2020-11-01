from pathlib import Path
import h5py
from prepare_data import *


folder_name = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15,
    17, 21, 22, 24, 25, 27, 28, 29, 30, 32, 34, 35,
    36, 38, 39, 40, 42, 43, 44
]


base_input_path = Path('shocks_rps')


ca_k = sp.profile(nx=33, ny=1, ns=4, nw=37)

f = h5py.File('observed_3.nc')

ca_k.wav[:] = f['wav'][()]

ick = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

f.close()

for index, rp in enumerate(folder_name):

    filename = base_input_path / 'observed_{}.nc'.format(rp)
    f = h5py.File(filename, 'r')
    ca_k.dat[0, 0, index] = f['profiles'][0, 0, 0]
    f.close()


ca_k.weights[:, :] = 1.e16


ca_k.weights[ick, 0] = 0.002


ca_k.weights[-1, 0] = 0.004


ca_k.write('observed_quiet.nc')
