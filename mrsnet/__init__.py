# mrsnet/__init__.py - MRSNet - init package
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

__all__ = ["analyse", "basis", "dataset", "model", "molecules", "spectrum", "train"]

version_info = (1,1,0)
__version__ = '.'.join(map(str, version_info))
