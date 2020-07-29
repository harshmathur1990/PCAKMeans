from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt


base_dir = '/Volumes/Harsh 9599771751/Oslo Work'


stic_files = [
    (
        base_dir + '/output_profs/cycle_1_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_1_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_2_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_2_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_3_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_3_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_4_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_4_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_5_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_5_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_6_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_6_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_7_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_7_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_6_cw_profs_3.nc',
        base_dir + '/model_atmos/cycle_6_cw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_7_cw_profs_3.nc',
        base_dir + '/model_atmos/cycle_7_cw_atmos_3.nc'
    )
]

nicole_files = [
    (
        base_dir + '/output_profs/cycle_1_N_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_1_N_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_2_N_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_2_N_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_3_N_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_3_N_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_4_N_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_4_N_sw_atmos_3.nc'
    ),
    (
        base_dir + '/output_profs/cycle_5_N_sw_profs_3.nc',
        base_dir + '/model_atmos/cycle_5_N_sw_atmos_3.nc'
    )
]

falc_file = base_dir + '/model_atmos/falc_nicole.nc'
observed_file = base_dir + '/rps/observed_3.nc'
outdir = Path(
    '/Users/harshmathur/CourseworkRepo/OsloAnalysis/InversionRPs/3/sw'
)


def plot_individual_plots():

    # plot profile
    for files in stic_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        f = h5py.File(files[0], 'r')
        fo = h5py.File(observed_file, 'r')

        indices = np.where(fo['profiles'][0, 0, 0, :-1, 0] != 0)[0]
        plt.plot(
            fo['wav'][:-1][indices],
            fo['profiles'][0, 0, 0, :-1, 0][indices],
            label='observed'
        )
        plt.plot(
            f['wav'][:-1],
            f['profiles'][0, 0, 0, :-1, 0],
            label='fit'
        )
        plt.title('Inverted vs Observed')
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Normalised Intensity')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[0]).name + '_profile.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        f.close()
        fo.close()

    for files in nicole_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        f = h5py.File(files[0], 'r')
        fo = h5py.File(observed_file, 'r')

        indices = np.where(fo['profiles'][0, 0, 0, :-1, 0] != 0)[0]
        plt.plot(
            fo['wav'][:-1][indices],
            fo['profiles'][0, 0, 0, :-1, 0][indices],
            label='observed'
        )
        plt.plot(
            f['wav'][:-1],
            f['profiles'][0, 0, 0, :-1, 0],
            label='fit'
        )
        plt.title('Inverted vs Observed')
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Normalised Intensity')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[0]).name + '_profile.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        f.close()
        fo.close()

    # plot temperature
    for files in stic_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['temp'][0][0][0],
            label='inverted temp'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['temp'][0][0][0],
            label='falc'
        )
        plt.title('FALC Temp vs Inverted Model Temperature')
        plt.xlabel('Log tau')
        plt.ylabel('Kelvin')
        plt.ylim(0, 18000)
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_temperature.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()

    for files in nicole_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['temp'][0][0][0],
            label='inverted temp'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['temp'][0][0][0],
            label='falc'
        )
        plt.title('FALC Temp vs Inverted Model Temperature')
        plt.xlabel('Log tau')
        plt.ylabel('Kelvin')
        plt.ylim(0, 18000)
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_temperature.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()

    # plot vlos
    for files in stic_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vlos'][0][0][0],
            label='inverted vlos'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['vlos'][0][0][0],
            label='falc'
        )
        plt.title('FALC Vlos vs Inverted Vlos')
        plt.xlabel('Log tau')
        plt.ylabel('Velocity (cm/sec)')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_vlos.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()

    for files in nicole_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vlos'][0][0][0],
            label='inverted vlos'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['vlos'][0][0][0],
            label='falc'
        )
        plt.title('FALC Vlos vs Inverted Vlos')
        plt.xlabel('Log tau')
        plt.ylabel('Velocity (cm/sec)')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_vlos.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()

    # plot vturb
    for files in stic_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vturb'][0][0][0],
            label='inverted vturb'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['vturb'][0][0][0],
            label='falc'
        )
        plt.title('FALC Vturb vs Inverted Vturn')
        plt.xlabel('Log tau')
        plt.ylabel('Velocity (cm/sec)')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_vturb.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()

    for files in nicole_files:
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos = h5py.File(files[1], 'r')
        falc = h5py.File(falc_file, 'r')

        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vturb'][0][0][0],
            label='inverted vturb'
        )
        plt.plot(
            falc['ltau500'][0][0][0],
            falc['vturb'][0][0][0],
            label='falc'
        )
        plt.title('FALC Vturb vs Inverted Vturb')
        plt.xlabel('Log tau')
        plt.ylabel('Velocity (cm/sec)')
        plt.legend()
        plt.tight_layout()
        outfilename = Path(files[1]).name + '_vturb.png'
        outfile_path = outdir / outfilename
        plt.savefig(outfile_path, format='png', dpi=300)
        plt.close('all')
        plt.clf()
        plt.cla()
        atmos.close()
        falc.close()


def plot_collective_comparison():

    # Plot profile
    plt.close('all')
    plt.clf()
    plt.cla()

    fo = h5py.File(observed_file, 'r')
    indices = np.where(fo['profiles'][0, 0, 0, :-1, 0] != 0)[0]

    plt.plot(
        fo['wav'][:-1][indices],
        fo['profiles'][0, 0, 0, :-1, 0][indices],
        label='observed'
    )

    for index, files in enumerate(stic_files):
        f = h5py.File(files[0], 'r')
        plt.plot(
            f['wav'][:-1],
            f['profiles'][0, 0, 0, :-1, 0],
            label='set {}'.format(index + 1)
        )
        f.close()

    plt.title('Inverted vs Observed')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Normalised Intensity')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_stic' + '_profile.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    fo.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    fo = h5py.File(observed_file, 'r')
    indices = np.where(fo['profiles'][0, 0, 0, :-1, 0] != 0)[0]

    plt.plot(
        fo['wav'][:-1][indices],
        fo['profiles'][0, 0, 0, :-1, 0][indices],
        label='observed'
    )

    for index, files in enumerate(nicole_files):
        f = h5py.File(files[0], 'r')
        plt.plot(
            f['wav'][:-1],
            f['profiles'][0, 0, 0, :-1, 0],
            label='set {}'.format(index + 1)
        )
        f.close()

    plt.title('Inverted vs Observed')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Normalised Intensity')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_nicole' + '_profile.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    fo.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    # plot temperature
    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['temp'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(stic_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['temp'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Temp vs Inverted Model Temperature')
    plt.xlabel('Log tau')
    plt.ylabel('Kelvin')
    plt.ylim(0, 18000)
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_stic' + '_temperature.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['temp'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(nicole_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['temp'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Temp vs Inverted Model Temperature')
    plt.xlabel('Log tau')
    plt.ylabel('Kelvin')
    plt.ylim(0, 18000)
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_nicole' + '_temperature.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    # plot vlos
    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['vlos'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(stic_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vlos'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Vlos vs Inverted Vlos')
    plt.xlabel('Log tau')
    plt.ylabel('Velocity (cm/sec)')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_stic' + '_vlos.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['vlos'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(nicole_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vlos'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Vlos vs Inverted Vlos')
    plt.xlabel('Log tau')
    plt.ylabel('Velocity (cm/sec)')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_nicole' + '_vlos.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()


    # plot vturb
    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['vturb'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(stic_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vturb'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Vturb vs Inverted Vturb')
    plt.xlabel('Log tau')
    plt.ylabel('Velocity (cm/sec)')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_stic' + '_vturb.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()

    plt.close('all')
    plt.clf()
    plt.cla()

    falc = h5py.File(falc_file, 'r')
    plt.plot(
        falc['ltau500'][0][0][0],
        falc['vturb'][0][0][0],
        label='falc'
    )

    for index, files in enumerate(nicole_files):
        atmos = h5py.File(files[1], 'r')
        plt.plot(
            atmos['ltau500'][0][0][0],
            atmos['vturb'][0][0][0],
            label='set {}'.format(index + 1)
        )
        atmos.close()

    plt.title('FALC Vturb vs Inverted Vturb')
    plt.xlabel('Log tau')
    plt.ylabel('Velocity (cm/sec)')
    plt.legend()
    plt.tight_layout()
    outfilename = 'sw_comparison_nicole' + '_vturb.png'
    outfile_path = outdir / outfilename
    plt.savefig(outfile_path, format='png', dpi=300)
    falc.close()
    plt.close('all')
    plt.clf()
    plt.cla()
