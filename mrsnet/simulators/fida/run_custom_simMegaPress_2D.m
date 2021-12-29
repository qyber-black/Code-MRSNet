% run_simMegaPressShapedEdit_2D.m
%
% SPDX-FileCopyrightText: Copyright (C) 2021  <lw1660@gmail.com> Swansea University
% SPDX-License-Identifier: BSD-3-Clause
%
% USAGE:
% Called by mrsnet.
%
% DESCRIPTION:
% Simulates a MEGA-PRESS experiment with shaped editing and refocusing pulses for
% for a finite-size voxel with phase cycling and spatial averaging.
%
% INPUTS: command line inputs
%
% Npts   number of spectral points
% sw     width of output spectrum (1/dwelltime) [Hz]
% lw     spectral linewidth [Hz]
% Bfield field strength [T]

% OUTPUTS: files only
%
% *MEGAPRESS_2D*      output mat file with full data  
% *MEGAPRESS_EDITOFF  output file with simulated MEGA-PRESS average edit-ON spectrum
% *MEGAPRESS_EDITON   output file with simulated MEGA-PRESS average edit-OFF spectrum

if ~exist(save_dir, 'dir')
  mkdir(save_dir);
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
p.taus          = [5 17 17 17 12];         % pulse sequence timings [ms]
p.centreFreq    = 3;

p.refocWaveform = 'sampleRefocPulse.pta';  % name of refocus pulse waveform
p.refTp         = 5;                       % duration of refocusing pulses[ms]
p.refPhCyc1     = [0,90];                  % phase cycling 1st refocusing pulse [degrees]
p.refPhCyc2     = [0,90];                  % phase cycling 2nd refocusing pulse [degrees]
p.thkX          = 3;                       % slice thickness x refocusing pulse [cm]
p.thkY          = 3;                       % slice thickness y refocusing pulse [cm]
p.x             = linspace(-2.5,2.5,8);    % X positions to simulate [cm]
p.y             = linspace(-2.5,2.5,8);    % y positions to simulate [cm]

p.editWaveform  = 'sampleEditPulse.pta';   % name of editing pulse waveform
p.editOnFreq    = 1.900000000000000;       % frequency of edit on  pulse [ppm]
p.editOffFreq   = 7.400000000000000;       % frequency of edit off pulse [ppm]
p.editTp        = 20;                      % duration of editing pulses [ms]
p.editPhCyc1    = [0 90];                  % phase cycling steps for 1st editing pulse [degrees]
p.editPhCyc2    = [0 90 180 270];          % phase cycling steps for 2nd editing pulse [degrees]
p.spinSys       = [];                      % spin system to simulate

% Load RF waveforms
editRF = io_loadRFwaveform(p.editWaveform,'inv',0);
refRF  = io_loadRFwaveform(p.refocWaveform,'ref',0);

% Resample refocusing RF pulse from 400 pts to 100 pts to reduce computational workload
p.refRF     = rf_resample(refRF,100);
p.editRFon  = rf_freqshift(editRF,p.editTp,(p.centreFreq-p.editOnFreq )*p.Bfield*gamma/1e6);
p.editRFoff = rf_freqshift(editRF,p.editTp,(p.centreFreq-p.editOffFreq)*p.Bfield*gamma/1e6);
p.Gx        = (refRF.tbw/(p.refTp/1000))/(gamma*p.thkX/10000); %[G/cm]
p.Gy        = (refRF.tbw/(p.refTp/1000))/(gamma*p.thkY/10000); %[G/cm]

totalIters = length(p.x)*length(p.y)*length(p.editPhCyc1)*length(p.editPhCyc2)*length(p.refPhCyc1)*length(p.refPhCyc2)

for ii = 1:length(metabolites)
    for jj = 1:length(linewidths)

      p.lw    = linewidths(jj);   % change lw default
      spinSys = metabolites{ii}; % spin system to simulate
      load spinSystems;          % load spin systems and select desired metabolite
      sys = eval(['sys' spinSys])

      % Run full simulation and save output in f_name
      f_name = sprintf('FIDA2D_%s_MEGAPRESS_%.2f_%d_%d_%.2f.mat',metabolites{ii}, p.lw, p.sw, npts, mrsnet_omega);
      [ON,OFF,DIFF] = SimMega2D(sys,p,f_name,save_dir);

      % Save average edit ON and OFF spectra separately
      % basic sequence parameters
      pulse_sequence = 'megapress';
      linewidth      = OFF.linewidth;
      omega_out      = OFF.Bo*gamma*1e-6;    % omega = gamma*Bo in MHz
      m_name         = metabolites{ii};   
      % fixed parameters
      t    = OFF.t;
      nu   = OFF.ppm;
      % edit off 
      edit = false;
      fid  = OFF.fids;
      fft  = OFF.specs;
      f_name = sprintf('FIDA2D_%s_MEGAPRESS_EDITOFF_%.2f_%d_%d_%.2f.mat',metabolites{ii}, p.lw, p.sw, npts, omega_out);
      save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'edit', 'pulse_sequence');
      % edit on
      edit = true;
      fid = ON.fids;
      fft = ON.specs;
      f_name = sprintf('FIDA2D_%s_MEGAPRESS_EDITON_%.2f_%d_%d_%.2f.mat',metabolites{ii}, p.lw, p.sw, npts, omega_out);
      save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'edit', 'pulse_sequence');
    end
end

