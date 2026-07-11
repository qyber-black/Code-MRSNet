% run_custom_simPress_2D.m
%
% SPDX-FileCopyrightText: Dana Goerzen and Jamie Near, McGill University 2021.
% SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
% SPDX-License-Identifier: BSD-3-Clause
%
% USAGE:
% Called by mrsnet.
%
% DESCRIPTION:
% Simulates a short-TE PRESS experiment (unedited) with shaped refocusing pulses
% for a finite-size voxel with spatial averaging. This is the PRESS analogue of
% run_custom_simMegaPress_2D.m: no editing pulses, no edit dimension, a single
% output sub-spectrum per metabolite/linewidth.
%
% INPUTS: command line inputs (as set by mrsnet)
%   npts   number of spectral points
%   sw     width of output spectrum (1/dwelltime) [Hz]
%   mrsnet_omega  spectrometer frequency [MHz] (gamma*Bfield)
%   metabolites   cell array of spin-system names
%   linewidths    vector of output linewidths [Hz]
%   save_dir      output directory
%
% OUTPUTS: files only
%   FIDA2D_<metab>_PRESS_ALL_...    output mat file with full 2D data
%   FIDA2D_<metab>_PRESS_PRESS_...  output file with the simulated PRESS spectrum

if ~exist(save_dir, 'dir')
  mkdir(save_dir);
end

% Optional switch: when true, cache unbroadened results and apply linewidths post hoc
if ~exist('use_cached_unbroadened','var') || isempty(use_cached_unbroadened)
  use_cached_unbroadened = false;
end

% Optional cache directory for unbroadened results
if ~exist('cache_dir','var') || isempty(cache_dir)
  cache_dir = fullfile(save_dir, 'cache_unbroadened');
end

gamma  = 42577000;                         % gyromagnetic ratio (approximate!)
% Create input structure
if exist('mrsnet_omega','var')
   p.Bfield = mrsnet_omega*1e6/gamma;                 % magnetic field strength [Tesla]
else  % defined such that gamma*Bfield = 123.2 MHz using FID-A value of gamma = 42577000;
   p.Bfield = 2.893649004133784;
end
if exist('npts','var')
   p.Npts = npts;
else
   p.Npts = 4096;                          % number of spectral points
end
if exist('sw','var')
  p.sw = sw;
else
  p.sw = 2000;                             % width of output spectrum [Hz]
end
% other defaults
p.lw            = 0.750000000000000;       % linewidth of output spectrum [Hz]
% Short-TE PRESS: TE = tau1 + tau2 = 35 ms (symmetric spin-echo timing)
p.tau1          = 17.5;                     % TE1 for first spin echo [ms]
p.tau2          = 17.5;                     % TE2 for second spin echo [ms]
p.centreFreq    = 3;                        % centre frequency of refocusing pulses [ppm]
p.flipAngle     = 180;                      % refocusing pulse flip angle [degrees]

p.refocWaveform = 'sampleRefocPulse.pta';  % name of refocus pulse waveform
p.refTp         = 5;                       % duration of refocusing pulses[ms]
p.thkX          = 3;                       % slice thickness x refocusing pulse [cm]
p.thkY          = 3;                       % slice thickness y refocusing pulse [cm]
p.x             = linspace(-2.5,2.5,8);    % X positions to simulate [cm]
p.y             = linspace(-2.5,2.5,8);    % y positions to simulate [cm]

p.spinSys       = [];                      % spin system to simulate

% Load RF waveform
refRF  = io_loadRFwaveform(p.refocWaveform,'ref',0);

% Resample refocusing RF pulse from 400 pts to 100 pts to reduce computational workload
p.refRF     = rf_resample(refRF,100);
p.Gx        = (refRF.tbw/(p.refTp/1000))/(gamma*p.thkX/10000); %[G/cm]
p.Gy        = (refRF.tbw/(p.refTp/1000))/(gamma*p.thkY/10000); %[G/cm]

totalIters = length(p.x)*length(p.y)

for ii = 1:length(metabolites)
    if use_cached_unbroadened
      % Use cache-based generator: simulate once unbroadened and apply all linewidths
      spinSys = metabolites{ii};
      load spinSystems;
      sys = eval(['sys' spinSys])
      m_name = metabolites{ii};
      omega  = p.Bfield*gamma*1e-6; % MHz
      % If caller provided basis_roots, forward to generator
      if exist('basis_roots','var') && ~isempty(basis_roots)
        generate_press2d_with_cache(sys, p, linewidths, m_name, omega, save_dir, cache_dir, basis_roots);
      else
        generate_press2d_with_cache(sys, p, linewidths, m_name, omega, save_dir, cache_dir);
      end
    else
      for jj = 1:length(linewidths)

        p.lw    = linewidths(jj);   % change lw default
        spinSys = metabolites{ii}; % spin system to simulate
        load spinSystems;          % load spin systems and select desired metabolite
        sys = eval(['sys' spinSys]);

        % Run full simulation and save output in f_name
        f_name = sprintf('FIDA2D_%s_PRESS_ALL_%.2f_%d_%d_%.2f.mat',metabolites{ii}, p.lw, p.sw, npts, mrsnet_omega);
        out = SimPress2D(sys,p,f_name,save_dir);

        % Save the PRESS spectrum (single, unedited sub-spectrum)
        pulse_sequence = 'press';
        linewidth      = out.linewidth;
        omega          = out.Bo*gamma*1e-6;    % omega = gamma*Bo in MHz
        m_name         = metabolites{ii};
        t    = out.t;
        nu   = out.ppm;
        fid  = out.fids;
        fft  = out.specs;
        f_name = sprintf('FIDA2D_%s_PRESS_PRESS_%.2f_%d_%d_%.2f.mat',metabolites{ii}, p.lw, p.sw, npts, omega);
        save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'pulse_sequence');
      end
    end
end
