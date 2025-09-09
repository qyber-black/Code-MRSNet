# mrsnet/grid.py - MRSNet - key-value grid
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Grid search utilities for MRSNet.

This module provides classes for managing parameter grids and iterating
over all combinations of parameter values for model selection.
"""

import numpy as np
import json

class Grid:
  """Key-value grid for parameter search.

  This class represents a grid of parameter values where each key
  maps to a list of possible values for that parameter.

  Attributes:
      values (dict): Dictionary mapping parameter names to lists of values
  """

  def __init__(self, values):
    """Initialize a parameter grid.

    Args:
        values (dict): Dictionary mapping parameter names to lists of values
    """
    self.values = values

  def __iter__(self):
    """Return iterator for the grid.

    Returns:
        GridIterator: Iterator over all parameter combinations
    """
    return GridIterator(self)

  def __str__(self):
    """Return string representation of the grid.

    Returns:
        str: Formatted string showing all parameter values
    """
    return "Grid:\n" + "\n".join(["  "+k+": "+str(self.values[k]) for k in self.values])

  @staticmethod
  def load (filename):
    """Load a grid from a JSON file.

    Args:
        filename (str): Path to JSON file containing grid values

    Returns:
        Grid: Loaded grid object
    """
    with open(filename, 'rb') as load_file:
      values = json.load(load_file)
      for k in values:
        for l in range(0,len(values[k])):
          if isinstance(values[k][l],list):
            values[k][l].sort()
      return Grid(values)

  @staticmethod
  def all_combinations_sort(lst):
    """Generate all sorted combinations of elements in a list.

    Args:
        lst (list): List of elements to combine

    Returns:
        list: List of all sorted combinations
    """
    from itertools import combinations
    res = []
    for l in range(1,len(lst)+1):
      res += [sorted(list(k)) for k in  combinations(lst,l)]
    return res

class GridIterator:
  """Iterator for grid combinations.

  This class provides iteration over all combinations of parameter values
  in a grid, returning one combination at a time.

  Attributes:
      _grid (Grid): The grid to iterate over
      _keys (list): List of parameter names
      _max_index (list): Maximum index for each parameter
      _index (numpy.ndarray): Current index for each parameter
      _max_level (int): Maximum level for iteration
  """

  def __init__(self, grid):
    """Initialize grid iterator.

    Args:
        grid (Grid): Grid object to iterate over
    """
    self._grid = grid
    self._keys = [k for k in self._grid.values.keys()]
    self._max_index = [len(self._grid.values[k]) for k in self._keys]
    self._index = np.zeros(len(self._keys),dtype=np.int64)
    self._max_level = len(self._keys)-1

  def __next__(self):
    """Get next parameter combination in the grid.

    Returns:
        list: Next combination of parameter values

    Raises:
        StopIteration: When all combinations have been exhausted
    """
    if self._max_level < 0:
      raise StopIteration
    res = [self._grid.values[self._keys[l]][self._index[l]] for l in range(0,len(self._keys))]
    self._index[self._max_level] += 1
    l = self._max_level
    while self._index[l] >= self._max_index[l]:
      self._index[l] = 0
      l -= 1
      if l < 0:
        self._max_level = -1
        break
      self._index[l] += 1
    return res
