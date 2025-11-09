import os
import glob
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import s_index_calculator as sic
import ispec

def determine_radial_velocity_with_mask(fname, ispec_dir):
    mu_cas_spectrum = ispec.read_spectrum(fname)
    #--- Radial Velocity determination with linelist mask --------------------------
    # - Read atomic data
    mask_file = os.path.join(ispec_dir, "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst")
    ccf_mask = ispec.read_cross_correlation_mask(mask_file)

    models, ccf = ispec.cross_correlate_with_mask(mu_cas_spectrum, ccf_mask, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, mask_depth=0.01, \
                            fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s

    return rv,rv_err


def correct_radial_velocity(fname,rv):
    mu_cas_spectrum = ispec.read_spectrum(fname)
    #--- Radial Velocity correction ------------------------------------------------
    mu_cas_spectrum = ispec.correct_velocity(mu_cas_spectrum, rv)

    return mu_cas_spectrum

def process_fits_files(directory, ispec_dir):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fits"):
                fname = os.path.join(root, file)
                sp = fits.open(fname)
                header = sp[0].header

                date = header['Date']
                bjd = header['OHP DRS BJD']
                star = str(header['OBJNAME'])
                instrumento = 'SOPHIE'

                rv,rv_err = determine_radial_velocity_with_mask(fname, ispec_dir)
                spectrum = correct_radial_velocity(fname,rv)

                wavelength = 10*spectrum['waveobs']
                flux = spectrum['flux']

                s_index_value = sic.S_index(wavelength,flux)

                results.append({
                    'Name': star,
                    'S_index': s_index_value,
                    'BJD': bjd,
                    'Date': date,
                    'RV': rv,
                    'RV_err': rv_err,
                    'Instrument': instrumento
                })
    return results
