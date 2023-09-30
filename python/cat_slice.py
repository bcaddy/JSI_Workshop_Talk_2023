#!/usr/bin/env python3
"""
Python script for concatenating slice hdf5 datasets for when -DSLICES is turned
on in Cholla. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_slice` function can be used to
concatenate the HDF5 files.

Generally the easiest way to import this script is to add the `python_scripts`
directory to your python path in your script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import cat_slice
```
"""

import h5py
import argparse
import pathlib

# ==============================================================================
def concat_slice(source_directory: pathlib.Path,
                 destination_file_path: pathlib.Path,
                 num_ranks: int,
                 timestep_number: int,
                 concat_xy: bool = True,
                 concat_yz: bool = True,
                 concat_xz: bool = True,
                 skip_fields: list = [],
                 destination_dtype: str = None,
                 compression_type: str = None,
                 compression_options: str = None):
  '''
  '''
  # Open destination file and first file for getting metadata
  source_file = h5py.File(source_directory / f'{timestep_number}_slice.h5.0', 'r')
  destination_file = h5py.File(destination_file_path, 'w')

  # Copy over header
  destination_file = copy_header(source_file, destination_file)

  # Get a list of all datasets in the source file
  datasets_to_copy = list(source_file.keys())

  # Filter the datasets to only include those I wish to copy
  if not concat_xy:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'xy' in dataset]
  if not concat_yz:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'yz' in dataset]
  if not concat_xz:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'xz' in dataset]
  datasets_to_copy = [dataset for dataset in datasets_to_copy if not dataset in skip_fields]

  # Create the datasets in the destination file
  for dataset in datasets_to_copy:
    dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

    slice_shape = get_slice_shape(source_file, dataset)

    destination_file.create_dataset(name=dataset,
                                    shape=slice_shape,
                                    dtype=dtype,
                                    compression=compression_type,
                                    compression_opts=compression_options)

  # Close source file in prep for looping through source files
  source_file.close()

  # Copy data
  for rank in range(num_ranks):
    # Open source file
    source_file = h5py.File(source_directory / f'{timestep_number}_slice.h5.{rank}', 'r')

    # Loop through and copy datasets
    for dataset in datasets_to_copy:
      # Determine locations and shifts for writing
      i0_start, i0_end, i1_start, i1_end = write_bounds(source_file, dataset)

      # Copy the data
      destination_file[dataset][i0_start:i0_end, i1_start:i1_end] = source_file[dataset]

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ==============================================================================

# ==============================================================================
def copy_header(source_file: h5py.File, destination_file: h5py.File):
  fields_to_skip = ['dims_local', 'offset']

  for attr_key in source_file.attrs.keys():
    if attr_key not in fields_to_skip:
      destination_file.attrs[attr_key] = source_file.attrs[attr_key]

  return destination_file
# ==============================================================================

# ==============================================================================
def get_slice_shape(source_file: h5py.File, dataset: str):
  nx, ny, nz = source_file.attrs['dims']

  if 'xy' in dataset:
    slice_dimensions = (nx, ny)
  elif 'yz' in dataset:
    slice_dimensions = (ny, nz)
  elif 'xz' in dataset:
    slice_dimensions = (nx, nz)
  else:
    raise ValueError(f'Dataset "{dataset}" is not a slice.')

  return slice_dimensions
# ==============================================================================

# ==============================================================================
def write_bounds(source_file: h5py.File, dataset: str):
  nx_local, ny_local, nz_local = source_file.attrs['dims_local']
  x_start, y_start, z_start    = source_file.attrs['offset']

  if 'xy' in dataset:
    bounds = (x_start, x_start+nx_local, y_start, y_start+ny_local)
  elif 'yz' in dataset:
    bounds = (y_start, y_start+ny_local, z_start, z_start+nz_local)
  elif 'xz' in dataset:
    bounds = (x_start, x_start+nx_local, z_start, z_start+nz_local)
  else:
    raise ValueError(f'Dataset "{dataset}" is not a slice.')

  return bounds
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  rootPath = pathlib.Path.home() /'Downloads'/'small_otv_test_data'
  start = default_timer()
  concat_slice(rootPath, rootPath/'outdir'/'output.h5', 16, 0)
  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')