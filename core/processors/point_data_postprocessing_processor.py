# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:19:13 2025

@author: Maksim Eremenko
"""
# processors/point_data_postprocessing_processor.py
import numpy as np
import os
import time
from dask import delayed, compute
from scipy.signal import convolve
from numba import jit
from scipy.fft import fft, fftshift
from scipy.signal import convolve
from scipy.signal.windows import  chebwin

class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        self.parameters = parameters

    def process_chunk(self, chunk_id, rifft_saver, output_dir):
        # Retrieve all point data associated with this chunk_id
        point_data_list = self.db_manager.get_point_data_for_chunk(chunk_id)
        if not point_data_list:
            print(f"No point data found for chunk_id: {chunk_id}")
            return None

        # Generate RIFFT grid and read amplitudes
        rifft_space_grid, amplitudes = self.load_amplitudes_and_generate_grid(chunk_id, point_data_list, rifft_saver)
        if rifft_space_grid.size == 0 or amplitudes is None:
            print(f"No valid data for chunk_id: {chunk_id}")
            return None

        # Process each grid around central points and calculate weighted averages
        self.calculate_and_save_positions(chunk_id, rifft_space_grid, amplitudes, output_dir)

    def load_amplitudes_and_generate_grid(self, chunk_id, point_data_list, rifft_saver):
        filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
        try:
            data = rifft_saver.load_data(filename)
            amplitudes = data.get('amplitudes', None)
            if amplitudes is None:
                print(f"Amplitudes not found in {filename}")
                return np.array([]), None

            # Generate grid points based on point data
            grids = []
            central_point_ids = []
            for pd in point_data_list:
                grid_points = self.point_data_processor._generate_grid(
                    chunk_id=chunk_id,
                    dimensionality=len(pd['coordinates']),
                    step_in_frac=pd['step_in_frac'],
                    central_point=pd['coordinates'],
                    dist=pd['dist_from_atom_center'],
                    central_point_id=pd['central_point_id']
                )
                grids.append(grid_points)
                central_point_ids.extend([pd['central_point_id']] * len(grid_points))

            rifft_space_grid = np.hstack((np.vstack(grids), np.array(central_point_ids)[:, None])) if grids else np.array([])
            return rifft_space_grid, amplitudes
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return np.array([]), None

    # Define filter windows based on dimensionality
    def filter_from_window(self, window0, dimensionality, window1=None, window2=None, size_aver=None):
        if dimensionality == 1:
            wd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (len(window0) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            win = wd0_s
        elif dimensionality == 2:
            wd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (len(window0) / 2.0)
            wd1 = fft(window1, np.array(self.parameters["supercell"])[1]) / (len(window1) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            wd1_s = np.abs(fftshift(wd1 / np.abs(wd1).max()))
            win = np.outer(wd0_s, wd1_s)
        elif dimensionality == 3:
            wd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (len(window0) / 2.0)
            wd1 = fft(window1, np.array(self.parameters["supercell"])[1]) / (len(window1) / 2.0)
            wd2 = fft(window2, np.array(self.parameters["supercell"])[2]) / (len(window2) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            wd1_s = np.abs(fftshift(wd1 / np.abs(wd1).max()))
            wd2_s = np.abs(fftshift(wd2 / np.abs(wd2).max()))
            
            # Initialize 3D kernel
            win = np.ones((np.array(self.parameters["supercell"])[0], np.array(self.parameters["supercell"])[1], np.array(self.parameters["supercell"])[2]))
            for i in range(np.array(self.parameters["supercell"])[0]):
                for j in range(np.array(self.parameters["supercell"])[1]):
                    for k in range(np.array(self.parameters["supercell"])[2]):
                        win[i, j, k] = wd0_s[i] * wd1_s[j] * wd2_s[k]
        else:
            raise ValueError("Unsupported dimensionality: {}".format(dimensionality))
        return win
    
    def calculate_and_save_positions(self, chunk_id, rifft_space_grid, amplitudes, output_dir):
        # Group by central_point_id
        unique_ids = np.unique(rifft_space_grid[:, -1])
        # Define the output file path
        output_file_path = os.path.join(output_dir, f"output_chunk_{chunk_id}.dat")
        
        with open(output_file_path, 'w') as file:
            self.write_header(file, dimensionality=rifft_space_grid.shape[1] - 1)
            dimensionality=rifft_space_grid.shape[1] - 1
            for central_point_id in unique_ids:
                mask = rifft_space_grid[:, -1] == central_point_id
                grid_points = rifft_space_grid[mask, :-1]
                r_positions_val_delta = np.real(amplitudes[mask])[:,1]
                # Select filter based on configuration
                filtWin = chebwin
                size_aver = np.array(self.parameters["supercell"])
                
                if dimensionality == 1:
                    window0 = filtWin(size_aver[0] // 2, 100)
                    filtWin = self.filter_from_window(window0, 1, size_aver=size_aver)
                   
                elif dimensionality == 2:
                    window0 = filtWin(size_aver[0] // 2, 100)
                    window1 = filtWin(size_aver[1] // 2, 100)
                    filtWin = self.filter_from_window(window0, 2, window1, size_aver=size_aver)

                    
                elif dimensionality == 3:
                    window0 = filtWin(size_aver[0] // 2, 100)
                    window1 = filtWin(size_aver[1] // 2, 100)
                    window2 = filtWin(size_aver[2] // 2, 100)
                    chebWin = self.filter_from_window(window0, 3, window1, window2, size_aver=size_aver)

                else:
                    raise ValueError("Unsupported dimensionality: {}".format(dimensionality))
                # Apply filtering
                if dimensionality == 1:
                    filtered_r = convolve(r_positions_val_delta, filtWin, mode='same')
                elif dimensionality == 2:
                    filtered_r = convolve(r_positions_val_delta.reshape(size_aver), filtWin, mode='same')
                    filtered_r = filtered_r.flatten()
                elif dimensionality == 3:
                    filtered_r = convolve(r_positions_val_delta.reshape(size_aver), filtWin, mode='same')
                    filtered_r = filtered_r.flatten()
                else:
                    filtered_r = r_positions_val_delta  # Fallback, should not occur
                    
                    
                    
                pos_max_coord, pos_max_weighted, pos_max, amplitudes_max_weighted = self.compute_weighted_positions(r_positions_val_delta, grid_points)
                pos_min_coord, pos_min_weighted, pos_min, amplitudes_min_weighted = self.compute_weighted_positions_min(r_positions_val_delta, grid_points)

                displ_vector = grid_points[pos_max_weighted] - grid_points[pos_min_weighted]

                if pos_max_coord is None or pos_min_coord is None:
                    continue


                file.write(
                   f"{int(central_point_id)} {pos_min_weighted} {pos_max_weighted} "
                   f"{' '.join(map(str, pos_max_coord))} "
                   f"{' '.join(map(str, pos_min_coord))} "
                   f"{' '.join(map(str, displ_vector))} "
                   f"{amplitudes_min_weighted} {amplitudes_max_weighted} "
                   f"{pos_min} {pos_max}\n"
                   )


    def write_header(self, file, dimensionality):
         header = [
             'central_point_id', 'pos_min_weighted', 'pos_max_weighted'
         ]
    
         coord_headers = [f'pos_max_coord_{dim}' for dim in range(dimensionality)]
         coord_headers += [f'pos_min_coord_{dim}' for dim in range(dimensionality)]
         coord_headers += [f'displacement_vector_{dim}' for dim in range(dimensionality)]
    
         header.extend(coord_headers)
         header += ['amplitude_min_weighted', 'amplitude_max_weighted', 'pos_min', 'pos_max']
    
         file.write(' '.join(header) + '\n')

    @staticmethod
    @jit(nopython=True)
    def compute_weighted_positions(r_positions_val_delta, r_space_grid, variable=0.0):
        max_per = variable * np.max(r_positions_val_delta)
        above_threshold = r_positions_val_delta > max_per

        if not np.any(above_threshold):
            return None, None, np.argmax(r_positions_val_delta), 0

        wghts = r_positions_val_delta[above_threshold] / np.sum(r_positions_val_delta[above_threshold])
        pos_max_coord = np.dot(wghts, r_space_grid[above_threshold])
        distances = np.sqrt(np.sum((pos_max_coord - r_space_grid) ** 2, axis=1))
        pos_max_weighted = np.argmin(distances)
        pos_max = np.argmax(r_positions_val_delta)
        amplitudes_max_weighted = np.dot(wghts, r_positions_val_delta[above_threshold])

        return pos_max_coord, pos_max_weighted, pos_max, amplitudes_max_weighted

    @staticmethod
    @jit(nopython=True)
    def compute_weighted_positions_min(r_positions_val_delta, r_space_grid, variable=0.0):
        min_per = variable * np.min(r_positions_val_delta)
        below_threshold = r_positions_val_delta <= min_per

        if not np.any(below_threshold):
            return None, None, np.argmin(r_positions_val_delta), 0

        wghts_av = r_positions_val_delta[below_threshold] / np.sum(r_positions_val_delta[below_threshold])
        pos_min_coord = np.dot(wghts_av, r_space_grid[below_threshold])
        distances = np.sqrt(np.sum((pos_min_coord - r_space_grid) ** 2, axis=1))
        pos_min_weighted = np.argmin(distances)
        pos_min = np.argmin(r_positions_val_delta)
        amplitudes_min_weighted = np.dot(wghts_av, r_positions_val_delta[below_threshold])

        return pos_min_coord, pos_min_weighted, pos_min, amplitudes_min_weighted
