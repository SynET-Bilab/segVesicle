import numpy as np
import gc
import scipy.fft


def deconv_tomo(vol, out_file, angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift, ncpu=8):
    data = np.arange(0, 1 + 1 / 2047., 1 / 2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data / highpassnyquist) * np.pi
    highpass = 1 - np.cos(highpass)
    eps = 1e-6
    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10 ** deconvstrength) * highpass + eps

    ctf = tom_ctf1d(angpix * 1e-10, voltage * 1e3, cs * 1e-3, -defocus * 1e-6, 0.07, phaseshift / 180 * np.pi, 0)
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf / (ctf * ctf + 1 / snr)

    s1 = - int(np.shape(vol)[1] / 2)
    f1 = s1 + np.shape(vol)[1] - 1
    m1 = np.arange(s1, f1 + 1)

    s2 = - int(np.shape(vol)[0] / 2)
    f2 = s2 + np.shape(vol)[0] - 1
    m2 = np.arange(s2, f2 + 1)

    s3 = - int(np.shape(vol)[2] / 2)
    f3 = s3 + np.shape(vol)[2] - 1
    m3 = np.arange(s3, f3 + 1)

    x, y, z = np.meshgrid(m1, m2, m3)
    x = x.astype(np.float32) / np.abs(s1)
    y = y.astype(np.float32) / np.abs(s2)
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    del x, y, z
    gc.collect()
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)

    ramp = np.interp(r, data, wiener).astype(np.float32)
    del r
    gc.collect()

    deconv = np.real(scipy.fft.ifftn(scipy.fft.fftn(vol, overwrite_x=True, workers=ncpu) * ramp, overwrite_x=True, workers=ncpu))
    deconv = deconv.astype(np.float32)
    std_deconv = np.std(deconv)
    std_vol = np.std(vol)
    ave_vol = np.average(vol)
    del vol, ramp
    gc.collect()

    deconv /= std_deconv
    deconv *= std_vol
    deconv += ave_vol
    gc.collect()

    return deconv

def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):

    ny = 1 / pixelsize


    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2


    points = np.arange(0,length)
    points = points.astype(float)
    points = points/(2 * length)*ny

    k2 = points**2
    term1 = lambda1**3 * cs * k2**2

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)


    return (pcurve + acurve)*bfactor