#! /usr/bin/env python

# Convert different S3 data inputs to the Eureka format

import glob, os
import numpy as np
import astraeus.xarrayIO as xrio
from copy import copy
import time as t

def convert_to_eureka(meta):
    '''Convert provided data into Eureka specific format

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object with values from the old MetaClass.
    spec : object
        Xarray Dataset containing saved information.
    '''
    # Get the input file name
    file = glob.glob(meta.inputdir+'*')

    # Remove already converted files from the list
    file = [f for f in file if '_EUREKACONVERT.' not in f]
    
    if len(file) == 0:
        raise ValueError(f'No files found in input directory {meta.inputdir}')
    elif len(file) > 1:
        raise ValueError('More than one file found in input directory'+
                          f'{meta.inputdir}, include only the data file.')
    else:
        file = file[0] #Take the first index because of glob returning list

    # Look for existing converted file:
    savefile_prefix = '.'.join(file.split('.')[:-1])
    savefile = savefile_prefix + '_EUREKACONVERT.h5'
    meta.filename_S3_SpecData = savefile
    if os.path.exists(savefile):
        spec = xrio.readXR(savefile)
    else:
        # Need to convert the file
        # Define the conversion function we want to use
        if meta.format == 'exotic_jedi':
            convert = convert_exotic_jedi
        elif meta.format == 'firefly':
            convert = convert_firefly
        elif meta.format == 'radica_soss':
            convert = convert_radica_soss

        # Call the conversion function on the file
        spec = convert(file, savefile)

    return meta, spec

def convert_exotic_jedi(file, savefile):
    '''Convert ExoTiC-JEDI data to Eureka! data

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    file : str
        String identifier for the data file
    savefile : str
        String identifier for where to save the converted file

    Returns
    -------
    spec : object
        Xarray Dataset containing saved information.
    '''
    raw_spec = xrio.readXR(file)

    # Get the data we want
    time = raw_spec['time_flux'].values
    wave = raw_spec['wavelength'].values
    specs = raw_spec['flux'].values
    err = raw_spec['flux_error'].values
    mask = 1-raw_spec['quality_flag'].values.astype(int)
    # Create x-axis array
    x = np.arange(len(wave))

    # Construct and save the dataset in Eureka! format
    spec = assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile)

    return spec

def convert_firefly(file, savefile):
    return

def convert_radica_soss(file, savefile):
    return

def assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile):
    '''Assemble a Eureka! compatible dataset

    Parameters
    ----------
    time : ndarray
        1D time array in BJD_TDB
    x : ndarray
        Array of pixels in the dispersion axis
    specs : ndarray
        2D array of extracted 1D spectra
    mask : ndarray
        2D array of x values to be masked
    wave : ndarray
        1D array of wavelengths

    Returns
    -------
    spec : object
        Xarray Dataset containing saved information.
    '''
    # Need to explicitly set unit labels
    # NOTE: If needed, better to adjust arrays to these units in the individual 
    # convert functions instead of messing with the unit labels
    time_units = 'BJD_TDB'
    wave_units = 'microns'
    flux_units = 'ELECTRONS'

    # Assemble the dataset
    spec = xrio.makeDataset()
    spec['time'] = time
    spec['x'] = x
    spec['optspec'] = (['time', 'x'], specs)
    spec['optspec'].attrs['flux_units'] = flux_units
    spec['optspec'].attrs['time_units'] = time_units
    spec['optspec'].attrs['wave_units'] = wave_units
    spec['opterr'] = (['time', 'x'], err)
    spec['opterr'].attrs['flux_units'] = flux_units
    spec['opterr'].attrs['time_units'] = time_units
    spec['opterr'].attrs['wave_units'] = wave_units
    spec['optmask'] = (['time', 'x'], mask)
    spec['optmask'].attrs['flux_units'] = 'None'
    spec['optmask'].attrs['time_units'] = time_units
    spec['optmask'].attrs['wave_units'] = wave_units
    spec['wave_1d'] = (['x'], wave)
    spec['wave_1d'].attrs['wave_units'] = 'microns'
    
    # Save this file so we don't need to repeat things
    xrio.writeXR(savefile, spec)

    return spec