#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 This contains the various shared tools that the python scripts will use
================================================================================
"""

import pathlib
import pickle
import subprocess
import os
import h5py
import numpy as np

# ==============================================================================
# Variables that are commonly used across many scripts
# ==============================================================================

# Repo Root Path
repo_root = pathlib.Path(__file__).resolve().parent.parent

# The path to the pickled dictionary
pickle_filepath = repo_root / 'python' / 'links.pkl'

# Path to data files
data_files_path = repo_root / 'data'

# The root GitHub URL
github_url_root = 'https://github.com/bcaddy/caddy-et-al-2023/blob'

# The titles of plots with their pretty formatting
pretty_names = {'density':'Density',
                'gas_pressure':'Gas Pressure',
                'energy':'Energy',
                'velocity_x':'$V_x$',
                'velocity_y':'$V_y$',
                'velocity_z':'$V_z$',
                'magnetic_x':'$B_x$',
                'magnetic_y':'$B_y$',
                'magnetic_z':'$B_z$',
                'magnetic_energy':'Magnetic Energy',
                'spec_kinetic':'Specific Kinetic Energy',
                'fast_magnetosonic':'Fast Magnetosonic Wave',
                'slow_magnetosonic':'Slow Magnetosonic Wave',
                'alfven_wave':r'Alfvén Wave',
                'mhd_contact_wave':'Entropy Wave',
                'standing':r'Standing Alfvén Wave',
                'moving':r'Traveling Alfvén Wave',
                'b&w':'Brio & Wu',
                'd&w':'Dai & Woodward',
                'einfeldt':'Einfeldt Strong Rarefaction',
                'rj1a':'Ryu & Jones 1a',
                'rj4d':'Ryu & Jones 4d',
                'b_squared_avg':r'Normalized $\left< B^2 \right>$ Evolution',
                'divergence':r'Divergence'}

# The colors for various quantities
colors = {'density':'blue', 'gas_pressure':'green', 'energy':'red',
          'velocity_x':'purple', 'velocity_y':'purple', 'velocity_z':'purple',
          'magnetic_x':'orange', 'magnetic_y':'orange', 'magnetic_z':'orange',
          'plmc':'red', 'ppmc':'blue'}


# ==============================================================================
# Various useful functions
# ==============================================================================
# ==============================================================================
def cholla_runner(exe_path: pathlib.Path = repo_root / 'cholla' / 'bin',
                  reconstructor: str = 'ppmc',
                  param_file_name: str = 'blank_settings_file.txt',
                  cholla_cli_args: str = '' ,
                  move_initial: bool = False,
                  move_final: bool = False,
                  initial_filename: str = 'initial',
                  final_filename: str = 'final') -> None:
    """Run cholla with the given parameters and organize/clear the resulting files

    Args:
        exe_path (pathlib.Path, optional): The path to the directory that contains the Cholla executable. Defaults to repo_root/'cholla'/'bin'.
        reconstructor (str, optional): The reconstructor to use. Allows 'ppmc' or 'plmc'. Defaults to 'ppmc'.
        param_file_name (str, optional): The name of the parameter file to use. Defaults to 'blank_settings_file.txt'.
        cholla_cli_args (str, optional): The CLI arguments to pass to Cholla. Defaults to ''.
        move_initial (bool, optional): If True move the initial conditions file, if false delete it. Defaults to False.
        move_final (bool, optional): If True move the final conditions file, if false delete it.. Defaults to False.
        initial_filename (str, optional): The new name of the initial conditions file. Defaults to 'initial'.
        final_filename (str, optional): The new name of the final conditions file. Defaults to 'final'.
    """

    if not (reconstructor == 'ppmc' or reconstructor == 'plmc'):
        raise ValueError("reconstructor argument can only be 'ppmc' or 'plmc'")

    # Generate Cholla run command
    cholla_path = exe_path / f'cholla.mhd.c3po.{reconstructor}'
    param_file_path = repo_root / 'python' / 'cholla-config-files' / param_file_name
    log_file = repo_root / 'cholla.log'

    command = f'{cholla_path} {param_file_path} {cholla_cli_args} >> {log_file} 2>&1'

    # Run Cholla
    os.system(command)

    # Move or delete data files
    data_source_dir = repo_root / 'python'
    initial_data = data_source_dir / '0.h5.0'
    final_data = data_source_dir / '1.h5.0'

    if move_initial:
        initial_data.rename(data_files_path / f'{initial_filename}.h5')
    else:
        initial_data.unlink()

    if move_final:
        final_data.rename(data_files_path / f'{final_filename}.h5')
    else:
        final_data.unlink()

    (data_source_dir / 'run_output.log').unlink()
    (data_source_dir / 'run_timing.log').unlink()
# ==============================================================================

# ==============================================================================
# This next section is all the functions for loading and managing the data
# ==============================================================================

# ==============================================================================
def load_conserved_data(file_name: str,
                        load_gamma: bool = False,
                        load_resolution: bool = False,
                        load_time: bool = False,
                        load_dx: bool = False) -> dict:
    """Load the conserved variables from the HDF5 file

    Args:
        file_name (str): The name of the HDF5 file, no extension
        load_gamma (bool, optional): Whether or not to load gamma. Defaults to False.
        load_resolution (bool, optional): Whether or not to load the resolution. Defaults to False.
        load_time (bool, optional): Whether or not to load the time of the snapshot. Defaults to False.
        load_dx (bool, optional): Whether or not to load the dx, dy, dz of the snapshot. Defaults to False.

    Returns:
        dict: The dictionary containing all the conserved data
    """
    file = h5py.File(data_files_path / f'{file_name}.h5', 'r')

    output = {}
    if load_resolution:
        output['resolution'] = file.attrs['dims']
    if load_gamma:
        output['gamma']      = file.attrs['gamma']
    if load_time:
        output['time']       = file.attrs['t']
    if load_dx:
        output['dx']         = file.attrs['dx']

    output['density']    = np.array(file['density'])
    output['energy']     = np.array(file['Energy'])
    output['momentum_x'] = np.array(file['momentum_x'])
    output['momentum_y'] = np.array(file['momentum_y'])
    output['momentum_z'] = np.array(file['momentum_z'])
    output['magnetic_x'] = np.array(file['magnetic_x'])
    output['magnetic_y'] = np.array(file['magnetic_y'])
    output['magnetic_z'] = np.array(file['magnetic_z'])

    return output
# ==============================================================================

# ==============================================================================
def center_magnetic_fields(data: dict) -> dict:
    """Compute the centered magnetic fields

    Args:
        data (dict): All the input data

    Returns:
        dict: The input data with new centered magnetic field data
    """
    data['magnetic_x_centered'] = 0.5 * (data['magnetic_x'][1:, :, :]
                                         + data['magnetic_x'][:-1, :, :])
    data['magnetic_y_centered'] = 0.5 * (data['magnetic_y'][:, 1:, :]
                                         + data['magnetic_y'][:, :-1, :])
    data['magnetic_z_centered'] = 0.5 * (data['magnetic_z'][:, :, 1:]
                                         + data['magnetic_z'][:, :, :-1])

    return data
# ==============================================================================

# ==============================================================================
def slice_data(data: dict,
               x_slice_loc: int = None,
               y_slice_loc: int = None,
               z_slice_loc: int = None) -> dict:
    """Slice all fields of the given data

    Args:
        data (dict): The given data
        x_slice_loc (int, optional): The x location to slice at. Defaults to None.
        y_slice_loc (int, optional): The y location to slice at. Defaults to None.
        z_slice_loc (int, optional): The z location to slice at. Defaults to None.

    Returns:
        dict: The sliced data
    """

    def limits(slice_loc, slice_dir, field):
        if slice_loc == None:
            low_lim = 0
            high_lim = field.shape[slice_dir]
        else:
            low_lim = slice_loc
            high_lim = low_lim+1
        return low_lim, high_lim

    for key in data:
        if key in ['resolution', 'gamma']:
            continue

        field = data[key]

        x_low, x_high = limits(x_slice_loc, 0, field)
        y_low, y_high = limits(y_slice_loc, 1, field)
        z_low, z_high = limits(z_slice_loc, 2, field)

        data[key] = data[key][x_low:x_high, y_low:y_high, z_low:z_high]
        data[key] = data[key].squeeze() # remove all dimensions of size 1

    return data
# ==============================================================================

# ==============================================================================
def compute_velocities(data: dict) -> dict:
    """Compute the velocities and add them to the data dictionary

    Args:
        data (dict): All the input data

    Returns:
        dict: The input data with new velocities fields
    """
    data['velocity_x'] = data['momentum_x'] / data['density']
    data['velocity_y'] = data['momentum_y'] / data['density']
    data['velocity_z'] = data['momentum_z'] / data['density']

    return data
# ==============================================================================

# ==============================================================================
def compute_derived_quantities(data: dict, gamma: float) -> dict:
    """Compute the various derived quantities and add them to the data dictionary.
    This includes the gas pressure, total pressure, specific kinetic energy,
    and magnetic pressure/energy

    Args:
        data (dict): All the input data
        gamma (float): The adiabatic index

    Returns:
        dict: The input data with new derived quantities fields
    """

    data['spec_kinetic']    = 0.5 * (data['velocity_x']**2 + data['velocity_y']**2 + data['velocity_z']**2)
    data['magnetic_energy'] = 0.5 * (data['magnetic_x_centered']**2 + data['magnetic_y_centered']**2 + data['magnetic_z_centered']**2)

    data['gas_pressure'] = (gamma - 1) * (data['energy'] - data['density'] * data['spec_kinetic'] - data['magnetic_energy'])
    data['total_pressure'] = data['gas_pressure'] + data['magnetic_energy']

    return data
# ==============================================================================
