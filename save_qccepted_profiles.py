from prepare_data import *
from pathlib import Path
import shutil
import h5py


write_dir = Path('/home/harsh/OsloAnalysis/accepted_rp_inversions/')
base_input_dir = Path('/home/harsh/OsloAnalysis/rp_quiet/plots_v1/')

def save_inversion_result(index, rp):

	f = h5py.File(base_input_dir / 'quiet_falc_cycle_1_t_5_vl_1_vt_4_atmos.nc', 'r')

	m = sp.model(nx=1, ny=1, nt=1, ndep=150)

	m.ltau[0,0,0,:] = f['ltau500'][0, 0, index]

	m.temp[0,0,0,:] = f['temp'][0, 0, index]

	m.pgas[0,0,0,:] = f['pgas'][0, 0, index]

	m.vturb[0,0,0,:] = f['vturb'][0, 0, index]

	m.vlos[0,0,0,:] = f['vlos'][0, 0, index]

	rp_write_dir = write_dir / 'rp_{}'.format(rp)

	shutil.copy(
		str(base_input_dir / 'nodes.txt'),
		str(rp_write_dir / 'nodes.txt')
	)

	shutil.copy(
		str(base_input_dir / 'plot_{}.png'.format(rp)),
		str(rp_write_dir / 'plot_{}.png'.format(rp))
	)

	m.write(rp_write_dir / 'rp_{}_inversion_result_falc_vlos_vturb_zero_initial.nc'.format(rp))

	f.close()
