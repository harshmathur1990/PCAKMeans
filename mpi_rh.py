import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python/')
# ys.path.insert(1, '/home/harsh/RH-Old/python/')
import enum
import os
import numpy as np
import h5py
import xdrlib
from mpi4py import MPI
from pathlib import Path
import tables as tb
import shutil
from helita.sim import multi
import subprocess
import rh
import time


# atmos_file = Path(
#     '/data/harsh/run_bifrost/Atmos/bifrost_en024048_hion_0_504_0_504_-500000.0_2000000.0.nc'
# )

atmos_file = Path(
    '/home/harsh/BifrostRun/bifrost_en024048_hion_0_504_0_504_-500000.0_2000000.0.nc'
)

# output_file = Path('/data/harsh/bifrost_supplementary_outputs_using_RH/output.nc')

output_file = Path('/home/harsh/BifrostRun/bifrost_supplementary_outputs_using_RH/output.nc')

# rh_run_base_dirs = Path('/data/harsh/run_bifrost_dirs')

rh_run_base_dirs = Path('/home/harsh/BifrostRun/run_bifrost_dirs')

stop_file = rh_run_base_dirs / 'stop'

sub_dir_format = 'process_{}'

input_filelist = [
    'molecules.input',
    'kurucz.input',
    'atoms.input',
    'keyword.input',
    'ray.input',
    'contribute.input'
]


def create_mag_file(
    Bx, By, Bz,
    write_path,
    shape=(127, )
):
    b_filename = 'MAG_FIELD.B'
    xdr = xdrlib.Packer()

    Babs = np.sqrt(
        np.add(
            np.square(Bx),
            np.add(
                np.square(By),
                np.square(Bz)
            )
        )
    )

    Binc = np.arccos(np.divide(Bz, Babs))

    Bazi = np.arctan2(By, Bx)

    xdr.pack_farray(
        np.prod(shape),
        Babs.flatten(),
        xdr.pack_double
    )
    xdr.pack_farray(
        np.prod(shape),
        Binc,
        xdr.pack_double
    )
    xdr.pack_farray(
        np.prod(shape),
        Bazi,
        xdr.pack_double
    )

    with open(write_path / b_filename, 'wb') as f:
        f.write(xdr.get_buffer())


def write_atmos_files(write_path, x, y):
    atmos_filename = 'Atmos1D.atmos'
    f = h5py.File(atmos_file, 'r')
    multi.watmos_multi(
        str(write_path / atmos_filename),
        f['temperature'][0, x, y],
        f['electron_density'][0, x, y] / 1e6,
        z=f['z'][0, x, y] / 1e3,
        vz=f['velocity_z'][0, x, y] / 1e3,
        # vturb=f['velocity_turbulent'][0, x, y] / 1e3,
        nh=f['hydrogen_populations'][0, :, x, y] / 1e6,
        id='Bifrost {} {}'.format(x, y),
        scale='height'
    )
    create_mag_file(
        Bx=f['B_x'][0, x, y],
        By=f['B_y'][0, x, y],
        Bz=f['B_z'][0, x, y],
        write_path=write_path
    )
    f.close()


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(x, y, read_path):

    cwd = os.getcwd()

    os.chdir(read_path)

    out = rh.readOutFiles(atoms=['H'])

    populations[:, x, y, :] = out.atmos.nH.T

    a_voigt[:, x, y, :] = out.damping_H.adamp

    # this is [lower-upper] as RH stores upper->lower in the indice [lower, upper]
    transition_list = [(0, 3), (0, 1), (0, 4), (0, 7), (1, 5), (3, 5), (3, 8), (3, 6)]

    # cularr = np.zeros((8, 127), dtype=np.float64)
    for indice, (ii, jj) in enumerate(transition_list):
        Cul[indice, x, y, :] = np.array([out.collrate_H.C_rates[kk].C[ii,jj] for kk in range(127)])

    wave_indices = [1220, 1241, 827, 821, 3332, 3484, 3422, 3444]

    for indice, wave_indice in enumerate(wave_indices):
        eta_c[indice, x, y, :] = np.array(out.opacity.opacity[wave_indice].chi)

    for indice, wave_indice in enumerate(wave_indices):
        eps_c[indice, x, y, :] = np.array(out.opacity.opacity[wave_indice].eta)

    job_matrix[x, y] = 1

    os.chdir(cwd)

    return Status.Work_done


def make_ray_file():

    os.chdir('/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/run_3')

    out = rh.readOutFiles()

    wave = np.array(out.spect.lambda0)

    indices = list()

    interesting_waves = [121.5668, 121.5673, 102.5722, 102.5721, 656.275, 656.290, 656.2851, 656.2867, 500]

    for w in interesting_waves:
        indices.append(
            np.argmin(np.abs(wave-w))
        )

    f = open('ray.input', 'w')

    f.write('1.00\n')
    f.write(
        '{} {}'.format(
            len(indices),
            ' '.join([str(indice) for indice in indices])
        )
    )
    f.close()


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    if output_file.exists():
        f = h5py.File(output_file, mode='a', driver='mpio', comm=MPI.COMM_WORLD, libver='latest')
        # f.swmr_mode = True
        job_matrix = f['job_matrix']
        populations = f['populations']
        a_voigt = f['a_voigt']
        Cul = f['Cul']
        eta_c = f['eta_c']
        eps_c = f['eps_c']

    else:
        f = h5py.File(output_file, mode='w', driver='mpio', comm=MPI.COMM_WORLD, libver='latest')
        # f.swmr_mode = True
        job_matrix = f.create_dataset("job_matrix", (504, 504), dtype=np.int64)
        populations = f.create_dataset(
           'populations',
            (13, 504, 504, 127), dtype=np.float64
        )
        a_voigt = f.create_dataset(
           'a_voigt',
            (8, 504, 504, 127), dtype=np.float64
        )
        Cul = f.create_dataset(
           'Cul',
            (8, 504, 504, 127), dtype=np.float64
        )
        eta_c = f.create_dataset(
            'eta_c',
            (8, 504, 504, 127), dtype=np.float64
        )
        eps_c = f.create_dataset(
            'eps_c',
            (8, 504, 504, 127), dtype=np.float64
        )
        job_matrix[:, :] = 0

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        start_x = int(sys.argv[1])
        end_x = int(sys.argv[2])
        start_y = int(sys.argv[3])
        end_y = int(sys.argv[4])

        x, y = np.where(job_matrix[start_x:end_x, start_y:end_y] == 0)

        x = x + start_x

        y = y + start_y

        for i in range(x.size):
            waiting_queue.add(i)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': (item, x[item], y[item])
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

        sys.stdout.write('Finished First Phase\n')

        while len(running_queue) != 0 or len(waiting_queue) != 0:

            if stop_work == False and stop_file.exists():
                stop_work = True
                waiting_queue = set()
                stop_file.unlink()

            status_dict = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=2,
                status=status
            )
            sender = status.Get_source()
            jobstatus = status_dict['status']
            item, xx, yy = status_dict['item']
            sys.stdout.write(
                'Sender: {} x: {} y: {} Status: {}\n'.format(
                    sender, xx, yy, jobstatus.value
                )
            )
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': (new_item, x[new_item], y[new_item])
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

    if rank > 0:
        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item, x, y = work_type['item']

            sub_dir_path = rh_run_base_dirs / 'runs' / 'process_{}'.format(rank)
            sub_dir_path.mkdir(parents=True, exist_ok=True)
            for input_file in input_filelist:
                shutil.copy(
                    rh_run_base_dirs / input_file,
                    sub_dir_path / input_file
                )

            commands = [
                'rm -rf *.dat',
                'rm -rf *.out',
                'rm -rf spectrum*',
                'rm -rf background.ray',
                'rm -rf Atmos1D.atmos',
                'rm -rf MAG_FIELD.B'
            ]

            start_time = time.time()
            for cmd in commands:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(sub_dir_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
                process.communicate()

            sys.stdout.write(
                'Rank: {} RH Remove Files Time: {}\n'.format(
                    rank, time.time() - start_time
                )
            )

            start_time = time.time()
            write_atmos_files(sub_dir_path, x, y)
            sys.stdout.write(
                'Rank: {} RH Make Atmosphere Files Time: {}\n'.format(
                    rank, time.time() - start_time
                )
            )

            # cmdstr = '/home/harsh/RH-Old/rhf1d/rhf1d'

            cmdstr = '/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/rhf1d'

            command = '{} 2>&1 | tee output.txt'.format(
                cmdstr
            )

            start_time = time.time()
            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            sys.stdout.write(
                'Rank: {} RH Run Time: {}\n'.format(
                    rank, time.time() - start_time
                )
            )

            start_time = time.time()
            status = do_work(x, y, sub_dir_path)
            sys.stdout.write(
                'Rank: {} RH Save Time: {}\n'.format(
                    rank, time.time() - start_time
                )
            )
            comm.send({'status': Status.Work_done, 'item': (item, x, y)}, dest=0, tag=2)

    f.close()
