import os
from astropy.io import fits
import numpy as np
import s_index_calculator as sic
import ispec
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count

def determine_radial_velocity_with_mask(fname, ispec_dir, spectral_type="G2"):
    """
    Determina a velocidade radial de um espectro usando o método de correlação cruzada
    com uma máscara de linhas.
    """
    sys.path.insert(0, os.path.abspath(ispec_dir))
    star_spectrum = ispec.read_spectrum(fname)
    mask_filename = f"HARPS_SOPHIE.{spectral_type}.375_679nm/mask.lst"
    mask_file = os.path.join(ispec_dir, "input/linelists/CCF", mask_filename)
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Máscara para o tipo espectral '{spectral_type}' não encontrada em: {mask_file}")
    ccf_mask = ispec.read_cross_correlation_mask(mask_file)
    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, ccf_mask, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, mask_depth=0.01, \
                            fourier=False)
    rv = np.round(models[0].mu(), 2)
    rv_err = np.round(models[0].emu(), 2)
    return rv, rv_err

def correct_radial_velocity(fname, rv):
    """
    Corrige a velocidade radial de um espectro.
    """
    star_spectrum = ispec.read_spectrum(fname)
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return star_spectrum

def process_file(fname, ispec_dir, spectral_type):
    """
    Processa um único arquivo de espectro (FITS ou TXT) para calcular o índice S.
    """
    try:
        if fname.endswith(".fits"):
            with fits.open(fname) as sp:
                header = sp[0].header
                required_keys = ['DATE', 'OHP DRS BJD', 'OBJNAME']
                if not all(key in header for key in required_keys):
                    return None, {'file': fname, 'error': f'Cabeçalho FITS incompleto. Faltando chaves: {[key for key in required_keys if key not in header]}'}

                date = header.get('DATE')
                bjd = header.get('OHP DRS BJD')
                star = header.get('OBJNAME')
                instrumento = 'SOPHIE'
                rv, rv_err = determine_radial_velocity_with_mask(fname, ispec_dir, spectral_type)
                spectrum = correct_radial_velocity(fname, rv)
                wavelength = 10 * spectrum['waveobs']
                flux = spectrum['flux']

        elif fname.endswith(".txt"):
            try:
                df = pd.read_csv(fname, delim_whitespace=True, names=['wavelength', 'flux'], dtype={'wavelength': np.float64, 'flux': np.float64})
                if df['wavelength'].isnull().any() or df['flux'].isnull().any():
                     return None, {'file': fname, 'error': 'Arquivo de texto contém valores não numéricos.'}
                wavelength = df['wavelength'].values
                flux = df['flux'].values
            except (ValueError, pd.errors.ParserError):
                return None, {'file': fname, 'error': 'Formato de arquivo de texto inválido. Esperado duas colunas numéricas.'}

            star = os.path.basename(fname).split('.')[0]
            date, bjd, instrumento, rv, rv_err = 'N/A', 'N/A', 'TXT', 0, 0
        else:
            return None, {'file': fname, 'error': 'Unsupported file type'}

        s_index_value = sic.S_index(wavelength, flux)

        result = {
            'Name': star, 'S_index': s_index_value, 'BJD': bjd, 'Date': date,
            'RV': rv, 'RV_err': rv_err, 'Instrument': instrumento
        }
        return result, None
    except Exception as e:
        return None, {'file': fname, 'error': str(e)}

def process_spectra_files(directory, ispec_dir, spectral_type):
    """
    Processa todos os arquivos FITS e TXT em um diretório e seus subdiretórios
    em paralelo para calcular o índice S.
    """
    files_to_process = [os.path.join(root, file)
                        for root, _, files in os.walk(directory)
                        for file in files if file.endswith((".fits", ".txt"))]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_file, [(fname, ispec_dir, spectral_type) for fname in files_to_process])

    successful_results = [res for res, err in results if res is not None]
    errors = [err for res, err in results if err is not None]

    return successful_results, errors
