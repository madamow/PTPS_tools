# !/usr/bin/python

import numpy as np
import scipy.interpolate as scin

cls = 299792.458

rejt_reg = np.array([
    [5256.1, 5256.5],
    [5599.2, 5599.7],
    [5612.7, 5613.2],
    [6085.5, 6085.9],
    [6376.5, 6377.2],
    [6620.1, 6620.8]])


def do_linear(spectrum):
    first = spectrum[0, 0]
    last = spectrum[-1, 0]

    res = (last-first) / spectrum.shape[0]
    x = np.arange(first, last, res)
    y = np.interp(x, spectrum[:, 0], spectrum[:, 1])

    out = np.transpose(np.vstack((x, y)))
    return out, res


def auto_rejt(tab):
    SN_tab = []
    for i, item in enumerate(rejt_reg):
        r_low, r_up = item
        rejt_area = tab[np.where((tab[:, 0] > r_low) & (tab[:, 0] < r_up))]
        if rejt_area.size > 2:
            reg = do_linear(rejt_area)[0]
            n = reg[:, 1].size
            avr = np.average(reg[:, 1])
            wy1 = reg[:, 1].max()-reg[:, 1].min()
            rms_tab = ((reg[:, 1]-avr)/wy1)**2
            rms = wy1 * np.sqrt(np.sum(rms_tab)/(n-1.))
            SN = np.average(reg[:, 1])/rms
            SN_tab.append(SN)
        else:
            continue
    if SN_tab:
        return np.average(SN_tab)
    else:
        return 100.


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


def wl_shift(lam, vel):
    lam_shifted = lam * np.sqrt((1.0+(vel/cls))/(1.0-(vel/cls)))
    return lam_shifted


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


def join_spec(spec):
    for ap in spec:
        if ap == 1:
            onel = spec[ap]['lam0']
            ones = spec[ap]['cont']
        else:
            last = spec[ap]['lam0'][-1]
            ind = abs(last-onel).argmin()

            onel = onel[ind:]
            ones = ones[ind:]

        onel = np.insert(onel, np.zeros(spec[ap]['lam0'].shape[0]),
                         spec[ap]['lam0'])
        ones = np.insert(ones, np.zeros(spec[ap]['lam0'].shape[0]),
                         spec[ap]['cont'])
    print(onel)
    exit()

    # join_tab=np.zeros((onel.shape[0],2))
    # for i in range(onel.shape[0]):
    #    join_tab[i,:]=[onel[i],ones[i]]

    # return join_tab
