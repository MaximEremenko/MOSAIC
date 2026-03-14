from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit
def corresponding_value(n_val, a):
    if n_val == 0:
        return abs(a)
    return np.sign(n_val) * (abs(n_val) + abs(a))


@njit
def unique_sorted_rounded(arr):
    for i in range(len(arr)):
        arr[i] = round(arr[i], 8)

    if len(arr) == 0:
        return arr
    count = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            count += 1
    result = np.empty(count, dtype=arr.dtype)
    idx = 0
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            idx += 1
            result[idx] = arr[i]
    return result


@njit
def find_val_in_interval(coord_min, coord_max, a, tol=1.0):
    coord_min_int = np.int32(np.floor(coord_min))
    coord_max_int = np.int32(np.ceil(coord_max))

    coord_min = round(coord_min, 8)
    coord_max = round(coord_max, 8)

    max_solutions = coord_max_int - coord_min_int + 4
    solutions = np.empty(max_solutions, dtype=np.float64)
    count = 0

    for n_val in range(coord_min_int - 2, coord_max_int + 2):
        if n_val == 0:
            a_abs = abs(a)
            neg_a = -a_abs
            pos_a = a_abs
            if (coord_min - 1 <= a <= coord_max + 1) and (
                coord_min - 1 <= -a <= coord_max + 1
            ):
                if (coord_min - tol) <= neg_a <= (coord_max + tol):
                    solutions[count] = neg_a
                    count += 1
                if (coord_min - tol) <= pos_a <= (coord_max + tol):
                    solutions[count] = pos_a
                    count += 1
            else:
                if coord_min <= pos_a <= coord_max:
                    solutions[count] = pos_a
                    count += 1
                if coord_min <= neg_a <= coord_max:
                    solutions[count] = neg_a
                    count += 1
        else:
            expr_val = corresponding_value(n_val, a)
            if coord_min <= expr_val <= coord_max:
                solutions[count] = expr_val
                count += 1

    solutions = solutions[:count]
    if count > 1:
        solutions = np.sort(solutions)
    return unique_sorted_rounded(solutions)

