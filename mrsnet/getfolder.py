# mrsnet/getfolder.py - MRSNet - get unique sub-folder for data storage
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""File system utilities for MRSNet.

This module provides utilities for creating unique subfolders for data storage
with file locking to prevent race conditions.
"""

import errno
import os
import time


def get_folder(folder, subfolder_pattern, timeout=30, delay=.1):
  """Get a unique subfolder for data storage.

  Creates a unique subfolder within the specified folder using a file lock
  to prevent race conditions when multiple processes try to create folders
  simultaneously.

  Parameters
  ----------
      folder (str): Base folder path
      subfolder_pattern (str): Pattern for subfolder names (e.g., "run-%s")
      timeout (int, optional): Timeout in seconds for acquiring lock. Defaults to 30
      delay (float, optional): Delay between lock attempts in seconds. Defaults to 0.1

  Returns
  -------
      str: Path to the created unique subfolder

  Raises
  ------
      RuntimeError: If lock acquisition times out or fails
  """
  # Lock
  os.makedirs(folder,exist_ok=True)
  lockfile = os.path.join(folder, "mrsnet.lock")
  locked = False
  start_time = time.time()
  while True:
    try:
      fd = os.open(lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)
      locked = True
      break
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
      if (time.time() - start_time) >= timeout:
        raise RuntimeError("get_folder timeout.") from e
      time.sleep(delay)
  if not locked:
    raise RuntimeError("Lock failed")

  # Create unique folder
  idl = 1
  while os.path.exists(os.path.join(folder,subfolder_pattern % str(idl))):
    idl = idl + 1
  subfolder = os.path.join(folder,subfolder_pattern % str(idl))
  os.makedirs(subfolder)

  # Unlock
  os.close(fd)
  os.unlink(lockfile)

  return subfolder
