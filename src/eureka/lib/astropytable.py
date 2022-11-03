from astropy.table import QTable
from astropy.io import ascii
import numpy as np


def savetable_S5(filename, time, wavelength, bin_width, lcdata, lcerr,
                 individual_models, model, residuals):
    """Save the results from Stage 5 as an ECSV file.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    time : ndarray (1D)
        The times for each data point.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    lcdata : ndarray (1D)
        The normalized flux measurements for each data point.
    lcerr : ndarray (1D)
        The normalized uncertainties for each data point.
    individual_models : ndarray (2D)
        An array containing pairs of model names and evaluated models.
    model : ndarray (1D)
        The predicted values from the fitted model.
    residuals : ndarray (1D)
        The residuals from lcdata - model.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    dims = [len(time), len(wavelength)]

    orig_shapes = [str(time.shape), str(wavelength.shape),
                   str(bin_width.shape), str(lcdata.shape), str(lcerr.shape),
                   str(individual_models.shape), str(model.shape),
                   str(residuals.shape)]

    time = np.tile(time, dims[1])
    wavelength = np.repeat(wavelength, dims[0])
    bin_width = np.repeat(bin_width, dims[0])
    lcdata = lcdata.flatten()
    lcerr = lcerr.flatten()
    model_names = individual_models[:, 0]
    model_values = individual_models[:, 1]
    full_model = model.flatten()
    residuals = residuals.flatten()

    arr = [time, wavelength, bin_width, lcdata, lcerr, *model_values,
           full_model, residuals]

    try:
        table = QTable(arr, names=('time', 'wavelength', 'bin_width',
                                   'lcdata', 'lcerr', *model_names, 'model',
                                   'residuals'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         "time, wavelength, bin_width, lcdata, lcerr, "
                         "individual_models, model, residuals\n"
                         ",".join(orig_shapes)) from e


def savetable_S6(filename, key, wavelength, bin_width, value, error):
    """Save the results from Stage 6 as an ECSV.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    key : str
        The parameter being saved.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    value : ndarray (1D)
        The fitted value at each wavelength.
    error : ndarray (1D)
        The uncertainty on each value.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    orig_shapes = [str(wavelength.shape), str(bin_width.shape),
                   str(value.shape), str(error[0].shape),
                   str(error[1].shape)]

    arr = [wavelength.flatten(), bin_width.flatten(), value.flatten(),
           error[0].flatten(), error[1].flatten()]

    try:
        table = QTable(arr, names=('wavelength', 'bin_width', key+'_value',
                                   key+'_errorneg', key+'_errorpos'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         f"wavelength, bin_width, {key}_value, {key}_errorneg,"
                         " {key}_errorpos\n"
                         ",".join(orig_shapes)) from e


def readtable(filename):
    """Read in a saved ECSV file.

    Parameters
    ----------
    filename : str
        The fully qualified filename of the file to read.

    Returns
    -------
    astropy.table.QTable
        The table previously saved by savetable_S5 or savetable_S6.
    """
    return ascii.read(filename, format='ecsv')
