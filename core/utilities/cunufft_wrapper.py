# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:11:22 2024

@author: Maksim Eremenko
"""

# utilities/cunufft_wrapper.py

import cufinufft
import cupy as cp
import numpy as np

def execute_cunufft(real_coords: np.ndarray, c: np.ndarray, q_coords: np.ndarray, eps=1e-12):
    """
    Executes the appropriate finufft nufftXd3 routine based on dimensionality.
    
    Args:
        real_coords (np.ndarray): Real-space coords (N, D), D=1,2,3
        c (np.ndarray): complex weights (N,)
        q_coords (np.ndarray): Reciprocal-space coords (M, D)
        eps (float): NUFFT accuracy parameter
    
    Returns:
        np.ndarray: NUFFT output (M,)
    """
    D = real_coords.shape[1]
    
    if D == 1:
        # x in real space, s in q space
        x = real_coords[:, 0].astype(np.float64, order='C')
        s = q_coords[:, 0].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft1d3(cp.asarray(x), 
                                             cp.asarray(c),
                                             cp.asarray(s), 
                                             eps=eps))
    elif D == 2:
        x = real_coords[:, 0].astype(np.float64, order='C')
        y = real_coords[:, 1].astype(np.float64, order='C')
        s = q_coords[:, 0].astype(np.float64, order='C')
        t = q_coords[:, 1].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft2d3(cp.asarray(x),
                                             cp.asarray(y), 
                                             cp.asarray(c), 
                                             cp.asarray(s), 
                                             cp.asarray(t),
                                             eps=eps))
    elif D == 3:
        x = real_coords[:,0].astype(np.float64, order='C')
        y = real_coords[:,1].astype(np.float64, order='C')
        z = real_coords[:,2].astype(np.float64, order='C')
        s = q_coords[:,0].astype(np.float64, order='C')
        t = q_coords[:,1].astype(np.float64, order='C')
        u = q_coords[:,2].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft3d3(cp.asarray(x),
                                             cp.asarray(y),
                                             cp.asarray(z),
                                             cp.asarray(c),
                                             cp.asarray(s),
                                             cp.asarray(t),
                                             cp.asarray(u), 
                                             eps=eps))
    else:
        raise ValueError(f"Unsupported dimensionality {D}. Only 1D,2D,3D supported.")

def execute_inverse_cunufft(q_coords: np.ndarray, c: np.ndarray, real_coords: np.ndarray, eps=1e-12):
    """
    Executes the inverse NUFFT based on dimensionality.
    
    Args:
        q_coords (np.ndarray): Reciprocal-space coords (N, D), D=1,2,3
        c (np.ndarray): Complex weights (N,)
        real_coords (np.ndarray): Real-space coords (M, D)
        eps (float): NUFFT accuracy parameter
    
    Returns:
        np.ndarray: NUFFT output (M,)
    """
    D = q_coords.shape[1]
    
    if D == 1:
        s = -q_coords[:, 0].astype(np.float64, order='C')
        x = real_coords[:, 0].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft1d3(cp.asarray(s), 
                                  cp.asarray(c),
                                  cp.asarray(x),
                                  eps=eps))
    elif D == 2:
        s = -q_coords[:, 0].astype(np.float64, order='C')
        t = -q_coords[:, 1].astype(np.float64, order='C')
        x = real_coords[:, 0].astype(np.float64, order='C')
        y = real_coords[:, 1].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft2d3(cp.asarray(s),
                                  cp.asarray(t),
                                  cp.asarray(c),
                                  cp.asarray(x),
                                  cp.asarray(y),
                                  eps=eps))
    elif D == 3:
        s = -q_coords[:, 0].astype(np.float64, order='C')
        t = -q_coords[:, 1].astype(np.float64, order='C')
        u = -q_coords[:, 2].astype(np.float64, order='C')
        x = real_coords[:, 0].astype(np.float64, order='C')
        y = real_coords[:, 1].astype(np.float64, order='C')
        z = real_coords[:, 2].astype(np.float64, order='C')
        return cp.asnumpy(cufinufft.nufft3d3(cp.asarray(s),
                                  cp.asarray(t),
                                  cp.asarray(u), 
                                  cp.asarray(c), 
                                  cp.asarray(x), 
                                  cp.asarray(y), 
                                  cp.asarray(z), 
                                  eps=eps))
    else:
        raise ValueError(f"Unsupported dimensionality {D}. Only 1D, 2D, 3D supported.")