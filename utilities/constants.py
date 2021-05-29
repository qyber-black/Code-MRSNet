#!/usr/bin/env python3
#
# utilities/constants.py - MRSNet - constants
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

NAA_REFERENCE = -2.01
CR_REFERENCE = -3.015
WATER_REFERENCE = -4.75             # Temperature dependant, avoid using if at all possible
GYROMAGNETIC_RATIO = 42.57747892    # 1H (MHz/T) : https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar

# Molecule names understood by MRNet; mapping long-name to range of short-names.
# Preferred short name is the first in the array, e.g. for myo-inositol 'MyI' will be chosen over 'mi'
MOLECULE_NAMES = {
        'N-Acetylaspartate': ['NAA'],
        'Creatine': ['Cr', 'cre'],
        'GABA': ['GABA'],
        'Choline-truncated': ['Cho'],
        'Glutamate': ['Glu'],
        'Glutamine': ['Gln'],
        'Glutathione': ['gsh'],
        'Glycine': ['Gly'],
        'Lactate': ['Lac'],
        'Myo-Inositol': ['MyI', 'mi', 'ins'],
        'N-Acetylaspartylglutamic': ['NAAG'],
        'NAAG-truncated-siemens': ['NAAG-SIE'],
        'Phosphocreatine': ['PCr', 'pch'],
        'Taurine': ['Tau'],
        'Water': ['H2O'],
        'DSS': ['DSS'],
        'Alanine': ['ala'],
        'Aspartate': ['asp'],
        'Scyllo-Inositol': ['scyllo']
}

# This is not the best idea for storing metadata about the scan data, but it works and is simple.
#
# It attaches data to spectra loaded from DICOM sources where the dictionary key (below) is somewhere in the filename.
# E.g for one of the E1 MEGA-PRESS phantom scans it has "GABA_SERIES_NAA_ONLY" in the file name. This ID acts as the
# ID for the group of spectra, it is required for MEGA-PRESS spectra to correctly group them.
#
# Additionally EDIT_ON, EDIT_OFF and DIFF are needed to identify what acquisition they are (see Spectra->load_dicom()
# for more information).

DICOM_METADATA ={
    'E1':{
        'GABA00_NAA_15mM_Cr_0mM': {'metabolite_names': ['n-acetylaspartate'], 'concentrations': [15.0]},
        'GABA00_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine'], 'concentrations': [15.0, 8.0]},
        'GABA01_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 0.5]},
        'GABA02_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.0]},
        'GABA03_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.6]},
        'GABA04_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.1]},
        'GABA05_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.6]},
        'GABA06_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 3.1]},
        'GABA07_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 4.1]},
        'GABA08_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 6.1]},
        'GABA09_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 8.1]},
        'GABA10_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 10.1]},
        'GABA11_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 11.7]}
    },
    'E2':{
        'GABA00_NAA_15mM_Cr_0mM': {'metabolite_names': ['n-acetylaspartate'], 'concentrations': [15.0]},
        'GABA00_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine'], 'concentrations': [15.0, 8.0]},
        'GABA01_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 0.5]},
        'GABA02_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.0]},
        'GABA03_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.5]},
        'GABA04_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.0]},
        'GABA05_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.5]},
        'GABA06_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 3.0]},
        'GABA07_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 4.0]},
        'GABA08_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 5.0]},
        'GABA09_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 6.0]},
        'GABA10_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 6.9]},
        'GABA11_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 7.9]},
        'GABA12_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 8.9]},
        'GABA13_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 9.9]},
        'GABA14_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 11.8]}
    },
    'E3':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 3, 12]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 3, 12]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 3, 12]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 3, 12]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 5, 3, 12]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 3, 12]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 7, 3, 12]},
        'GABA08_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 7.9, 3, 12]},
        'GABA09_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8.9, 3, 12]},
        'GABA10_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 9.9, 3, 12]},
        'GABA11_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10.8, 3, 12]},
        'GABA12_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 11.8, 3, 12]},
        'GABA13_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12.8, 3, 12]},
        'GABA14_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 13.7, 3, 12]}
    },
    'E4a':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 3, 12]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 3, 12]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 3, 12]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 3, 12]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 3, 12]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 3, 12]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 3, 12]}
    },
    'E4b':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 3, 12]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 3, 12]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 3, 12]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 3, 12]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 3, 12]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 3, 12]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 3, 12]}
    },
    'E4c':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 3, 12]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 3, 12]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 3, 12]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 3, 12]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 3, 12]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 3, 12]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 3, 12]}
    },
    'E4d':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 3, 12]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 3, 12]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 3, 12]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 3, 12]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 3, 12]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 3, 12]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 3, 12]}
    },
    # Blank Example
    # 'Spectra_set_name':{
    #     'Some_Spectra_ID00': {},
    #     'Some_Spectra_ID01': {}
    # }
}
