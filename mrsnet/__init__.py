# mrsnet/__init__.py - MRSNet - init package
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

__all__ = ["analyse", "basis", "cfg", "cnn", "compare", "dataset",
           "grid", "molecules", "selection", "spectrum", "train"]

version_info = (2,0,0)
__version__ = '.'.join(map(str, version_info))
