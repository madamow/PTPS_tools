import os
import numpy as np
from scipy.stats import pearsonr
import scipy.interpolate as scin

cls = 299792.458

def wl_shift(lam, vel):
    lam_shifted = lam * np.sqrt((1.0+(vel/cls))/(1.0-(vel/cls)))
    return lam_shifted

def read_lambda(fits, apertures):

    header = fits[1].header
    spec = fits[1].data[:]
    ciag = ""
    WATstring = []

    for card in fits[0].header.cards['WAT2_*']:
        if len(card[1]) == 67:
            ciag += ("%s " % card[1])
        else:
            ciag += card[1]

    for elem in ciag.split("="):
        for item in elem.split("spec"):
            if (len(item) > 6.0):
                WATstring.append(item.strip().strip("\""))

    spectrum = {}
    if WATstring:
        WATtab = np.array([item.split() for item in WATstring], dtype=float)

        # Pixels to wavelength transformation
        for ap in apertures:
            ap_flux = spec[ap - 1]
            # Apply Chebyshev polynomial transformation
            if (WATtab[ap, 2] == 2):
                x = np.linspace(-1, 1, len(ap_flux))
                coef = WATtab[ap - 1, 15:]
                lam = np.polynomial.chebyshev.chebval(x, coef)
            # Apply linear fit
            elif WATtab[ap - 1, 2] == 0:
                first = WATtab[ap - 1, 3]
                step = WATtab[ap - 1, 4]
                lam = first + np.arange(len(ap_flux)) * step
            spectrum[ap] = {'lam': lam, 'flux': ap_flux}

    else:
        x = np.arange(spec.shape[0])+1.
        crval = fits[1].header['CRVAL1']
        cdelt = fits[1].header['CDELT1']
        try:
            ref = float(fits[1].header['CRPIX1'])
        except KeyError:
            ref = 1.

        try:
            cd11 = fits[1].header['CD1_1']
        except KeyError:
            cd11 = cdelt

        lam = crval + cd11 * (x - ref)
        spectrum[1] = {'lam': lam, 'flux': spec}

    return header, spectrum


def cont_snr_in(knots, to_fit):
    ksigma = []
    xc = []
    kn_nr = len(knots)
    for j, knot in enumerate(knots):
        if j == 0:
            bk = to_fit[np.where(to_fit[:, 0] < knot)]
        elif j == kn_nr-1:
            bk = to_fit[np.where(to_fit[:, 0] > knot)]
        else:
            bk = to_fit[np.where((to_fit[:, 0] > knot) &
                                 (to_fit[:, 0] < knots[j + 1]))]

        ksigma.append(1./(1.-np.std(bk[:, 1])))
        xc.append(np.average(bk[:, 0]))
    return xc, ksigma


def fit_cont(lam_flux, kn_nr, low, high):
    clean = False
    to_fit = np.zeros((len(lam_flux['lam']), 2))

    to_fit[:, 0] = lam_flux['lam']
    to_fit[:, 1] = lam_flux['flux']

    chitab = []
    while not clean:
        x, y = to_fit[:, 0], to_fit[:, 1]
        odst = (x.max() - x.min())/(kn_nr+1.)
        knots = x[0]+odst*(np.arange(1, kn_nr+1.))

        if knots[-1] > x[-1]:
            knots[-1] = x[-1]-0.1

        xc, ksigma = cont_snr_in(knots, to_fit)
        s = scin.LSQUnivariateSpline(x, y, knots, k=3)
        ys = s(x)
        sigma = np.std(y/s(x))
        chi2 = np.average((y-ys)**2/ys)
        chitab.append(chi2)

        low_level = y/ys-low*sigma
        # build low level threshold
        for j, knot in enumerate(knots):
            if j == 0:
                ind = np.where(x[:] < knot)
            else:
                ind = np.where((x[:] < knot) & (x[:] > knots[j-1]))
            low_level[ind] = low_level[ind]*ksigma[j]

        high_level = ys + ys*high*sigma
        ind = np.where((y > low*s(x)) & (y < high_level))
        to_fit = np.transpose(np.array((x[ind], y[ind])))

        if (
            (len(chitab) > 1 and chitab[-2]-chitab[-1] < 0.0001)
            or to_fit.shape[0] < kn_nr
        ):
            clean = True

    cont = lam_flux['flux']/s(lam_flux['lam'])
    return cont


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

