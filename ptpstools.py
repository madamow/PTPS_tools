import json
import os
from readFITS import read_lambda, wl_shift, fit_cont
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def correlated_samples_with_errors(x, y, x_err, y_err, n_simulations=1000):
    """
    Calculates the distribution of Pearson correlation coefficients 
    for data with errors using Monte Carlo simulation.

    Args:
        x, y: arrays of measured values
        x_err, y_err: arrays of measurement errors for x and y
        n_simulations: number of simulations to run

    Returns:
        A list of correlation coefficients from each simulation.
    """
    correlations = []
    for _ in range(n_simulations):
        # Generate perturbed data points within error bounds
        x_sim = np.random.normal(x, x_err)
        y_sim = np.random.normal(y, y_err)
        
        # Calculate Pearson correlation coefficient
        corr, _ = pearsonr(x_sim, y_sim)
        correlations.append(corr)
    return correlations

def get_ptps_data(star):
    with open('PTPS_DB.json') as f:
        ptps = json.load(f)

    for record in ptps:
        if record.get('PTPS') == star or record.get('TYC') == star:
            return record  # Return immediately when found

    return None


def get_spectra(spec, data_dir, tel, epochs=None, zeroshift=True,
                cont=True, hetpart='red', aper=None):
    import astropy.io.fits as pyfits

    # If epochs keyword is not specified, plot all spectra from dict
    if epochs is None:
        epochs = [s['file'] for s in spec.get(tel, {}).get('RV', [])]

    if aper is None:
        aper = list(range(1, 24)) if tel == 'HET' else [1]

    if tel == 'TNG':
        fits_suffix = "_s1d_A.fits.fz"
    elif tel == 'HET' and hetpart == 'blue':
        fits_suffix = "_2.ms.fits.fz"
    else:
        fits_suffix = "_1.ms.fits.fz"

    rv_offset = spec.get('HET', {}).get('gc0RV', 0) if tel == 'HET' else 0

    spec_collection = {}
    #rv_dict = {e['file']: e for e in spec.get(tel, {}).get('RV', [])}
    rv_list = spec.get(tel, {}).get('RV', [])
    rv_list = sorted(rv_list, key=lambda x: x['mjd'])


    for epoch in epochs:
        e = next((d for d in rv_list if d['file'] == epoch), None)
  
        if not e:
            print("Epoch not found")
            continue  # Skip if the epoch is not found
    
        path = os.path.join(data_dir, epoch + fits_suffix)

        try:
            f1 = pyfits.open(path)
        except FileNotFoundError:
            print(f"Warning: File not found {path}")
            continue

        header, sout = read_lambda(f1, aper)

        # Apply continuum normalization
        if cont:
            rejt = (
                1 - (1 / e.get("snr", 50))
                if tel == "HET"
                else 1 - (1 / e.get("sn57", 50))
            )
            for ap in aper:
                try:
                    sout[ap]["cont"] = fit_cont(
                        sout[ap],
                        10 if tel == "HET" else 100,
                        rejt, 3
                    )
                except ValueError:
                    sout[ap]["cont"] = np.repeat(1., len(sout[ap]["lam"]))


        # Apply wavelength shift corrections
        if zeroshift:
            rv = rv_offset + (e['RV'] * 0.001 if tel == 'HET' else e['RV'])
            bc_stump = float(header.get('BC_STUMP', 0))
            for ap in aper:
                sout[ap]['lam0'] = wl_shift(sout[ap]['lam'], bc_stump)
                sout[ap]['lam0'] = wl_shift(sout[ap]['lam0'], -rv)

        spec_collection[epoch] = sout

    return spec_collection


sp = get_ptps_data('PTPS_1433')
s = get_spectra(sp, './spectra', 'TNG', hetpart='red', aper=[1], cont=True )
print(s)
exit()

for i, epoch in enumerate(s):
    for ap in s[epoch]:
        plt.plot(s[epoch][ap]['lam0'], s[epoch][ap]['cont']+(i*0.1), label=epoch)
#plt.legend()
#plt.axvline(5889.95)
#plt.axvline(5895.92)
#plt.axvline(6562.81)
#plt.axvline(6717.69)
#plt.xlim(6717.00,6718)
#plt.xlim(3900,4000)
plt.show()
