from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift


def filter_from_window(parameters, window0, dimensionality, window1=None, window2=None):
    if dimensionality == 1:
        fd0 = fft(window0, np.array(parameters["supercell"])[0]) / (len(window0) / 2.0)
        k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        return k0 / k0.sum()
    if dimensionality == 2:
        fd0 = fft(window0, parameters["supercell"][0]) / (len(window0) / 2.0)
        fd1 = fft(window1, parameters["supercell"][1]) / (len(window1) / 2.0)
        k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
        kern = k0[:, None] * k1[None, :]
        return kern / kern.sum()
    if dimensionality == 3:
        sc = np.array(parameters["supercell"])
        fd0 = fft(window0, sc[0]) / (len(window0) / 2.0)
        fd1 = fft(window1, sc[1]) / (len(window1) / 2.0)
        fd2 = fft(window2, sc[2]) / (len(window2) / 2.0)
        k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
        k2 = np.abs(fftshift(fd2 / np.abs(fd2).max()))
        kern = k0[:, None, None] * k1[None, :, None] * k2[None, None, :]
        return kern / kern.sum()
    raise ValueError("Unsupported dimensionality")
