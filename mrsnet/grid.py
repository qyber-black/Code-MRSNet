# mrsnet/grid.py - MRSNet - key-value grid
#
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

class Grid:
  def __init__(self, values):
    self.values = values

  def __iter__(self):
    return GridIterator(self)

  def __str__(self):
    return "Grid:\n" + "\n".join(["  "+k+": "+str(self.values[k]) for k in self.values])

  @staticmethod
  def all_combinations_sort(lst):
    from itertools import combinations
    res = []
    for l in range(1,len(lst)+1):
      res += [sorted(list(k)) for k in  combinations(lst,l)]
    return res

class GridIterator:
  def __init__(self, grid):
    self._grid = grid
    self._keys = [k for k in self._grid.values.keys()]
    self._max_index = [len(self._grid.values[k]) for k in self._keys]
    self._index = np.zeros(len(self._keys),dtype=np.int64)
    self._max_level = len(self._keys)-1

  def __next__(self):
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
