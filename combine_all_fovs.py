import h5py
import numpy as np
from pathlib import Path

file_1_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_662_712_708_758/plots/consolidated_results_velocity_calibrated_fov_662_712_708_758.h5'
)

file_2_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_rest_8_retry/plots/consolidated_results_velocity_calibrated_fov_rest_8_retry.h5'
)

file_3_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_535_585_715_765/plots/consolidated_results_velocity_calibrated_fov_535_585_715_765.h5'
)

label_file = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
)

write_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/'
)


def combine_all_fovs():
    f = h5py.File(file_1_path, 'r')
    fl = h5py.File(label_file, 'r')
    all_profiles = f['all_profiles'][4:11]
    syn_profiles = f['syn_profiles'][4:11]
    all_temp = f['all_temp'][4:11]
    all_vlos = f['all_vlos'][4:11]
    all_vturb = f['all_vturb'][4:11]
    all_labels = fl['new_final_labels'][4:11][:, 662:712, 708:758]
    fl.close()
    f.close()
    f = h5py.File(file_2_path, 'r')
    all_profiles = np.vstack([all_profiles, f['all_profiles'][()]])
    syn_profiles = np.vstack([syn_profiles, f['syn_profiles'][()]])
    all_temp = np.vstack([all_temp, f['all_temp'][()]])
    all_vlos = np.vstack([all_vlos, f['all_vlos'][()]])
    all_vturb = np.vstack([all_vturb, f['all_vturb'][()]])
    all_labels = np.vstack([all_labels, f['all_labels'][()]])
    f.close()
    f = h5py.File(file_3_path, 'r')
    all_profiles = np.vstack([all_profiles, f['all_profiles'][()]])
    syn_profiles = np.vstack([syn_profiles, f['syn_profiles'][()]])
    all_temp = np.vstack([all_temp, f['all_temp'][()]])
    all_vlos = np.vstack([all_vlos, f['all_vlos'][()]])
    all_vturb = np.vstack([all_vturb, f['all_vturb'][()]])
    all_labels = np.vstack([all_labels, f['all_labels'][()]])
    f.close()
    f = h5py.File(write_path / 'FoVAtoJ.nc', 'w')
    f['all_profiles'] = all_profiles
    f['syn_profiles'] = syn_profiles
    f['all_temp'] = all_temp
    f['all_vlos'] = all_vlos
    f['all_vturb'] = all_vturb
    f['all_labels'] = all_labels
    f.close()


if __name__ == '__main__':
    combine_all_fovs()
