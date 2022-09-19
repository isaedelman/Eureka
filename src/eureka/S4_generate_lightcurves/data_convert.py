#! /usr/bin/env python

# Convert different S3 data inputs to the Eureka format

import glob, os
import numpy as np
import astraeus.xarrayIO as xrio
from astropy.io import fits
from copy import copy

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
        elif meta.format == 'eureka_nometa':
            convert = convert_eureka_nometa

        # Call the conversion function on the file
        spec = convert(meta, file, savefile)

    return meta, spec

def convert_exotic_jedi(meta, file, savefile):
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

    #Gather drifts
    xcen = raw_spec['x_shift'].values
    ycen = raw_spec['y_shift'].values

    # Construct and save the dataset in Eureka! format
    spec = assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile,
                                    xcen=xcen, ycen=ycen)

    return spec

def convert_firefly(meta, file, savefile):
    '''Convert FireFly PRISM data to Eureka! data

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

    data = np.load(file, allow_pickle=True).item()
    # Assign things
    time = data['time']
    wave = data['wavelength']
    specs = data['specphot']
    err = data['specphot_err']    
    # Create x-axis array
    x = np.arange(len(wave))
    # Create a empty mask to satisfy Eureka!
    mask = np.zeros_like(specs)

    #Gather drifts
    xcen = data['shx']
    ycen = data['shy']

    # Construct and save the dataset in Eureka! format
    spec = assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile, \
                                    xcen=xcen, ycen=ycen)

    return spec

def convert_radica_soss(meta, file, savefile):
    '''Convert Michael Radica's SOSS data to Eureka! data

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
    with fits.open(file) as hdu:
        print(hdu.info())
        # Grab time
        time = hdu['Time'].data - 2400000.5

        if hasattr(meta, 'soss_order'):
            if meta.soss_order not in [1,2]:
                raise ValueError('Specified NIRISS SOSS order must either be '+
                    '"1" or "2"')
        else:
            meta.soss_order = 1
        print(f'Extracting Order {meta.soss_order} of NIRISS SOSS data')

        # Grab the Order info
        wave_lo = hdu['Wave Low O{}'.format(meta.soss_order)].data[0]
        wave_up = hdu['Wave Up O{}'.format(meta.soss_order)].data[0]
        wave = (wave_lo + wave_up) / 2
        specs = hdu['Flux O{}'.format(meta.soss_order)].data
        err = hdu['Flux Err O{}'.format(meta.soss_order)].data
    
    # Create x-axis array
    x = np.arange(len(wave))
    # Create a empty mask to satisfy Eureka!
    mask = np.zeros_like(specs)

    # Fix NaN's so they don't break the code
    specs[np.where(np.isnan(specs))] = 0
    err[np.where(np.isnan(specs))] = 0
    mask[np.where(np.isnan(specs))] = 1

    # All arrays are backwards, so lets flip them
    wave = np.flip(wave)
    specs = np.flip(specs, axis=1)
    err = np.flip(err, axis=1)

    # # Construct and save the dataset in Eureka! format
    spec = assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile)

    return spec 

def convert_eureka_nometa(meta, file, savefile):
    '''Wrapper to read in Eureka! data when no meta data was provided / saved

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
    spec = xrio.readXR(file)

    return spec 

def assemble_eureka_dataset(time, x, specs, err, mask, wave, savefile, \
                            xcen=np.array(False), ycen=np.array(False)):
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
    spec['time'].attrs['time_units'] = time_units
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
    spec['wave_1d'].attrs['wave_units'] = wave_units

    if xcen.any():
        spec['centroid_x'] = (['time'], xcen)
    if ycen.any():
        spec['centroid_y'] = (['time'], ycen)

    # Save this file so we don't need to repeat things
    xrio.writeXR(savefile, spec)

    return spec