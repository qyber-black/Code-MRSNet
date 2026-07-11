% run_custom_simPressShaped.m
%
% SPDX-FileCopyrightText: Dana Goerzen and Jamie Near, McGill University 2021.
% SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
% SPDX-License-Identifier: BSD-3-Clause
%
% USAGE:
% Called by mrsnet.
%
% DESCRIPTION:
% Simulates a short-TE PRESS experiment (unedited) with ideal refocusing pulses
% and no spatial averaging. This is the 1D PRESS analogue of
% run_custom_simMegaPressShapedEdit.m: no editing pulses, no edit dimension, a
% single output sub-spectrum per metabolite/linewidth.
%
% INPUTS: command line inputs (as set by mrsnet)
%   npts, sw, mrsnet_omega, metabolites, linewidths, save_dir
%
% OUTPUTS: files only
%   FIDA_<metab>_PRESS_PRESS_...  simulated PRESS spectrum

if ~exist(save_dir, 'dir')
  mkdir(save_dir);
end
for ii=1:length(metabolites)
    for jj=1:length(linewidths)
      spinSys=metabolites{ii}; %spin system to simulate

      gamma=42577000; %gyromagnetic ratio Hz/T

      % ************INPUT PARAMETERS**********************************
      Bfield = mrsnet_omega*1e6/gamma; %magnetic field strength [Tesla]
      lw     = linewidths(jj); %linewidth of the output spectrum [Hz]
      % Short-TE PRESS: TE = tau1 + tau2 = 35 ms (symmetric spin-echo timing)
      tau1   = 17.5; %TE1 for first spin echo [ms]
      tau2   = 17.5; %TE2 for second spin echo [ms]
      % ************END OF INPUT PARAMETERS**********************************

      %Load spin systems
      load spinSystems
      sys=eval(['sys' spinSys]);

      out = sim_press(npts,sw,Bfield,lw,sys,tau1,tau2);

      nu = out.ppm;
      pulse_sequence = 'press';
      linewidth = out.linewidth;
      t = out.t;
      omega = out.Bo * gamma * 1e-6;
      m_name = metabolites{ii};

      fid = out.fids;
      fft = out.specs;
      f_name = sprintf('FIDA_%s_PRESS_PRESS_%.2f_%d_%d_%.2f.mat',m_name, lw, sw, npts, omega);
      save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'pulse_sequence');
    end
end
