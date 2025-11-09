import os
from astropy.io import fits
import numpy as np
import s_index_calculator as sic
import ispec
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count

def determine_radial_velocity_with_mask(fname, ispec_dir):
    """
    Determina a velocidade radial de um espectro usando o método de correlação cruzada
    com uma máscara de linhas.

    Parâmetros:
    fname (str): Caminho para o arquivo do espectro (FITS ou txt).
    ispec_dir (str): Caminho para o diretório de instalação do iSpec.

    Retorna:
    tuple: Uma tupla contendo a velocidade radial (rv) e seu erro (rv_err).
    """
    # Adiciona o diretório do iSpec ao path do sistema para permitir a importação.
    sys.path.insert(0, os.path.abspath(ispec_dir))

    # Lê o espectro usando a função do iSpec.
    star_spectrum = ispec.read_spectrum(fname)

    # Define o caminho para a máscara de correlação cruzada.
    mask_file = os.path.join(ispec_dir, "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst")
    ccf_mask = ispec.read_cross_correlation_mask(mask_file)

    # Realiza a correlação cruzada para determinar a velocidade radial.
    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, ccf_mask, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, mask_depth=0.01, \
                            fourier=False)

    # Extrai a velocidade radial e seu erro do modelo ajustado.
    rv = np.round(models[0].mu(), 2)  # em km/s
    rv_err = np.round(models[0].emu(), 2)  # em km/s

    return rv, rv_err

def correct_radial_velocity(fname, rv):
    """
    Corrige a velocidade radial de um espectro.

    Parâmetros:
    fname (str): Caminho para o arquivo do espectro.
    rv (float): Valor da velocidade radial a ser corrigido.

    Retorna:
    dict: O espectro com a velocidade radial corrigida.
    """
    star_spectrum = ispec.read_spectrum(fname)
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return star_spectrum

def process_file(fname, ispec_dir):
    """
    Processa um único arquivo de espectro (FITS ou TXT) para calcular o índice S.
    """
    try:
        if fname.endswith(".fits"):
            with fits.open(fname) as sp:
                header = sp[0].header
                date = header.get('DATE', 'N/A')
                bjd = header.get('OHP DRS BJD', 'N/A')
                star = header.get('OBJNAME', 'N/A')
                instrumento = 'SOPHIE'
                rv, rv_err = determine_radial_velocity_with_mask(fname, ispec_dir)
                spectrum = correct_radial_velocity(fname, rv)
                wavelength = 10 * spectrum['waveobs']
                flux = spectrum['flux']
        elif fname.endswith(".txt"):
            df = pd.read_csv(fname, delim_whitespace=True, names=['wavelength', 'flux'])
            wavelength = df['wavelength'].values
            flux = df['flux'].values
            star = os.path.basename(fname).split('.')[0]
            date = 'N/A'
            bjd = 'N/A'
            instrumento = 'TXT'
            rv, rv_err = 0, 0
        else:
            return None, {'file': fname, 'error': 'Unsupported file type'}

        s_index_value = sic.S_index(wavelength, flux)

        result = {
            'Name': star,
            'S_index': s_index_value,
            'BJD': bjd,
            'Date': date,
            'RV': rv,
            'RV_err': rv_err,
            'Instrument': instrumento
        }
        return result, None
    except Exception as e:
        return None, {'file': fname, 'error': str(e)}

def process_spectra_files(directory, ispec_dir):
    """
    Processa todos os arquivos FITS e TXT em um diretório e seus subdiretórios
    em paralelo para calcular o índice S.
    """
    files_to_process = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".fits") or file.endswith(".txt"):
                files_to_process.append(os.path.join(root, file))

    # Use o número de CPUs disponíveis para o processamento paralelo.
    with Pool(cpu_count()) as pool:
        # Mapeia a função 'process_file' para cada arquivo a ser processado.
        # A função 'starmap' é usada para passar múltiplos argumentos para 'process_file'.
        results = pool.starmap(process_file, [(fname, ispec_dir) for fname in files_to_process])

    successful_results = [res for res, err in results if res is not None]
    errors = [err for res, err in results if err is not None]

    return successful_results, errors
