import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


write_path = Path('/data/harsh/merge_bifrost_output')
write_path.mkdir(parents=True, exist_ok=True)

read_path_format = '/data/harsh/bifrost_supplementary_outputs_using_RH/pixel_{}_{}/data.h5'


def merge_data():
    job_matrix = np.zeros((504, 504), dtype=np.int64)

    x, y = np.where(job_matrix == 0)

    t = tqdm(zip(x, y), total=x.size)

    populations = np.zeros((13, 504, 504, 127), dtype=np.float64)

    a_voigt = np.zeros((11, 504, 504, 127), dtype=np.float64)

    Cul = np.zeros((36, 504, 504, 127), dtype=np.float64)

    eta_c = np.zeros((11, 504, 504, 127), dtype=np.float64)

    eps_c = np.zeros((11, 504, 504, 127), dtype=np.float64)

    for index, (xx, yy) in enumerate(t):
        read_path = Path(read_path_format.format(xx, yy))

        f = h5py.File(read_path, 'r')

        populations[:, xx, yy, :] = f['populations'][()]

        a_voigt[:, xx, yy, :] = f['a_voigt'][()]

        Cul[:, xx, yy, :] = f['cularr'][()]

        eta_c[:, xx, yy, :] = f['eta_c'][()]

        eps_c[:, xx, yy, :] = f['eps_c'][()]

        f.close()

        t.set_postfix(processed=float(index)/float(x.size))

    fo = h5py.File(write_path / 'bifrost_output.h5', 'w')

    fo['populations'] = populations

    fo['a_voigt'] = a_voigt

    fo['Cul'] = Cul

    fo['eta_c'] = eta_c

    fo['eps_c'] = eps_c

    fo.close()


if __name__ == '__main__':
    merge_data()
