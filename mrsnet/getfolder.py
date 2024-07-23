# mrsnet/getfolder.py - MRSNet - get unique sub-folder for data storage
#
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import time
import errno

def get_folder(folder, subfolder_pattern, timeout=30, delay=.1):
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
        raise RuntimeError("get_folder timeout.")
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
