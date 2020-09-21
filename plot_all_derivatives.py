import h5py
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


base_path = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/no_vt_initial/custom_nodes/plots_v10'
)

derivative_filename = 'v10_falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_response.nc'

derivative_file = base_path / derivative_filename

f = h5py.File(derivative_file, 'r')

falc_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/model_atmos/falc_nicole_for_stic.nc'
)

falc = h5py.File(falc_file, 'r')

X, Y = np.meshgrid(f['wav'][:-1], falc['ltau500'][0][0][0])

derivatives_output = base_path / 'responses'

os.makedirs(derivatives_output, exist_ok=True)

for i in range(45):
    this_directory_path = derivatives_output / 'Response {}'.format(i)

    os.makedirs(this_directory_path, exist_ok=True)

    plt.close('all')
    plt.clf()
    plt.cla()
    plt.contourf(
        X,
        Y,
        f['derivatives'][0, 0, i, 0, :, :-1, 0],
        levels=1000,
        cmap='gray'
    )
    plt.title('Temperature Response Function')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Log Tau')
    plt.colorbar()
    plt.savefig(
        this_directory_path / 'temp_response.png',
        format='png',
        dpi=1200
    )
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.close('all')
    plt.clf()
    plt.cla()
    plt.contourf(
        X,
        Y,
        f['derivatives'][0, 0, i, 1, :, :-1, 0],
        levels=1000,
        cmap='gray'
    )
    plt.title('Velocity LOS Response Function')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Log Tau')
    plt.colorbar()
    plt.savefig(
        this_directory_path / 'vlos_response.png',
        format='png',
        dpi=1200
    )
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.close('all')
    plt.clf()
    plt.cla()
    plt.contourf(
        X,
        Y,
        f['derivatives'][0, 0, i, 2, :, :-1, 0],
        levels=1000,
        cmap='gray'
    )
    plt.title('Microturbulence Response Function')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Log Tau')
    plt.colorbar()
    plt.savefig(
        this_directory_path / 'vturb_response.png',
        format='png',
        dpi=1200
    )
    plt.close('all')
    plt.clf()
    plt.cla()

f.close()
falc.close()
