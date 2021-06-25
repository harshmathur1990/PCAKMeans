import sys
import h5py
import numpy as np
from pathlib import Path
from helita.sim import rh15d
from prepare_data import *
from witt import *
from scipy.integrate import cumtrapz

nicole_file = '/Users/harshmathur/CourseworkRepo/NICOLE/run/falc.model'
rh_file = '/Users/harshmathur/CourseworkRepo/rh/rh/Atmos/FALC_82_5x5.hdf5'
tau_dir = '/Users/harshmathur/CourseworkRepo/old_rh_runs/FALC_Ca_K_PRD_SST_WAVE'
pascal_to_dyne_per_cm_sq = 10
per_m_cu_to_per_cm_cu = 1e-6
kg_per_m_cu_to_g_per_cm_cu = 1e-3
boltzmann_constant = 1.380649e-23  # JK-1
m_per_sec_to_cm_per_sec = 1e2


w = witt()

vec_pg_from_pe = np.vectorize(w.pg_from_pe)


def write_stic_atmos(write_path=Path('.')):
    rh_atmos = h5py.File(rh_file, 'r')
    data = rh15d.Rh15dout(fdir=tau_dir)
    height = data.atmos.height_scale[0, 0].dropna('height')
    index500 = np.argmin(np.abs(data.ray.wavelength_selected - 500))
    tau500 = cumtrapz(data.ray.chi[0, 0, :, index500].dropna('height'), x=-height)
    tau500 = np.concatenate([[1e-20], tau500])

    electron_pressure = rh_atmos['electron_density'][0, 0, 0] * boltzmann_constant * rh_atmos['temperature'][0, 0, 0]
    gas_pressure = vec_pg_from_pe(rh_atmos['temperature'][0, 0, 0], electron_pressure)

    ######################-stic model-###################################################

    stic_filename = write_path / 'falc_rh_for_stic.nc'

    m = sp.model(nx=1, ny=1, nt=1, ndep=tau500.size)

    m.ltau[0,0,0,:] = np.log(tau500)

    m.temp[0,0,0,:] = rh_atmos['temperature'][0, 0, 0]

    m.pgas[0,0,0,:] = gas_pressure * pascal_to_dyne_per_cm_sq

    m.vturb[0,0,0,:] = rh_atmos['velocity_turbulent'][0, 0, 0] * m_per_sec_to_cm_per_sec

    m.vlos[0,0,0,:] = rh_atmos['velocity_z'][0, 0, 0] * m_per_sec_to_cm_per_sec

    m.write(str(stic_filename))

    ######################################################################################

    ############## NICOLE model ##########################################################

    nicole_filename = write_path / 'falc_rh_for_nicole.model'

    model = np.zeros((82, 8))

    model[:, 0] = np.log(tau500)

    model[:, 1] = rh_atmos['temperature'][0, 0, 0]

    model[:, 2] = electron_pressure * pascal_to_dyne_per_cm_sq

    model[:, 3] = rh_atmos['velocity_turbulent'][0, 0, 0] * m_per_sec_to_cm_per_sec

    model[:, 5] = rh_atmos['velocity_z'][0, 0, 0] * m_per_sec_to_cm_per_sec

    np.savetxt(nicole_filename, model)

    ######################################################################################

    #################  NiCOLE FALC TO STIC ###############################################

    nicole_falc = np.loadtxt(open(nicole_file).readlines()[2:])

    gas_pressure_nicole = vec_pg_from_pe(nicole_falc[:, 1], nicole_falc[:, 2] / pascal_to_dyne_per_cm_sq)[::-1]

    falc_nicole_for_stic = write_path / 'falc_nicole_for_stic.nc'

    m = sp.model(nx=1, ny=1, nt=1, ndep=nicole_falc.shape[0])

    m.ltau[0,0,0,:] = nicole_falc[:, 0][::-1]

    m.temp[0,0,0,:] = nicole_falc[:, 1][::-1]

    m.pgas[0,0,0,:] = gas_pressure_nicole * pascal_to_dyne_per_cm_sq

    m.vturb[0,0,0,:] = nicole_falc[:, 3][::-1]

    m.vlos[0,0,0,:] = nicole_falc[:, 5][::-1]

    m.write(str(falc_nicole_for_stic))
    ######################################################################################
