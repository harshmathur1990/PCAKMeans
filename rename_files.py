import os
from pathlib import Path


def rename():
    here = Path('.')

    all_files = here.glob('**/*')

    interesting_files = [file for file in all_files if 'profile_rps' in file.name]

    for interesting_file in interesting_files:
        tples = interesting_file.name.split('profile_rps')

        new_name = tples[0] + 'profiles_rps' + tples[1]

        os.rename(interesting_file.name, new_name)

if __name__ == '__main__':
    rename()
