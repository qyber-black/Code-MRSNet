% run_simMegaPressShapedEdit.m
%
% SPDX-FileCopyrightText: Jamie Near, McGill University 2014.
% SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
% SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
% SPDX-License-Identifier: BSD-3-Clause
%
% USAGE:
% This script is run simply by editing the input parameters and then
% clicking "Run".
%
% DESCRIPTION:
% This script simulates a MEGA-PRESS experiment with fully shaped editing
% pulses.  Phase cycling of editing pulses is performed, and summation
% across phase cycles is performed.
%
% INPUTS:
% To run this script, edit the parameters below as desired and then click
% "run":
% editWaveform          = name of editing pulse waveform.
% editOnFreq            = freqeucny of edit on pulse[ppm]
% editOffFreq           = frequency of edit off pulse[ppm]
% editTp                = duration of editing pulses[ms]
% npts                  = number of spectral points
% sw                    = spectral width [Hz]
% Bfield                = magnetic field strength [Tesla]
% lw                    = linewidth of the output spectrum [Hz]
% taus                  = vector of pulse sequence timings  [ms]
% spinSys               = spin system to simulate
% editPhCyc1            = vector of phase cycling steps for 1st editing pulse [degrees]
% editPhCyc2            = vector of phase cycling steps for 2nd editing pulse [degrees]
%
% OUTPUTS:
% outON             = Simulated MEGA-PRESS edit-ON spectrum.
% outOFF            = Simulated ExTE-MEGA-PRESS edit-OFF spectrum.
% outDIFF           = Simulated ExTE-MEGA-PRESS difference spectrum.

if ~exist(save_dir, 'dir')
  mkdir(save_dir);
end
for ii=1:length(metabolites)
    for jj=1:length(linewidths)
      spinSys=metabolites{ii}; %spin system to simulate

      gamma=42577000; %gyromagnetic ratio Hz/T

      % ************INPUT PARAMETERS**********************************
      editWaveform = 'sampleEditPulse.pta'; %name of editing pulse waveform.
      editOnFreq   = 1.9; %freqeucny of edit on pulse[ppm]
      editOffFreq  = 7.4; %frequency of edit off pulse[ppm]
      editTp       = 20; %duration of editing pulses[ms]
      % npts         = 4096; %number of spectral points
      % sw           = 1250; %spectral width [Hz]
      Bfield       = mrsnet_omega*1e6/gamma; %magnetic field strength [Tesla]
      lw           = linewidths(jj); %linewidth of the output spectrum [Hz]
      taus=[5 ,...  %Time from excitation to 1st refoc pulse [ms]
            17,...  %Time from 1st refoc pulse to 1st editing pulse [ms]
            17,...  %Time from 1st editing pulse to 2nd refoc pulse [ms]
            17,...  %Time from 2nd refoc pusle to 2nd editing pulse [ms]
            12];    %Time from 2nd editing pulse to ADC onset [ms]
      centreFreq  = 3.0; %Centre Frequency of MR spectrum [ppm];
      editPhCyc1  = [0 90]; %phase cycling steps for 1st editing pulse [degrees]
      editPhCyc2  = [0 90 180 270]; %phase cycling steps for 2nd editing pulse [degrees]
      % ************END OF INPUT PARAMETERS**********************************

      %Load RF waveforms
      editRF=io_loadRFwaveform(editWaveform,'inv',0);

      %Load spin systems
      load spinSystems
      sys=eval(['sys' spinSys]);

      %This is the step where the editing pulse waveform (initially a pulse with
      %zero-frequency) is frequency shifted to produce and edit-on and an
      %edit-off pulse;
      editRFon=rf_freqshift(editRF,editTp,(centreFreq-editOnFreq)*Bfield*gamma/1e6);
      editRFoff=rf_freqshift(editRF,editTp,(centreFreq-editOffFreq)*Bfield*gamma/1e6);

      %Initialize structures:
      outON_epc=cell(length(editPhCyc1),length(editPhCyc2));
      outOFF_epc=cell(length(editPhCyc1),length(editPhCyc2));
      outON=struct([]);
      outOFF=struct([]);

      for EP1=1:length(editPhCyc1)
          for EP2=1:length(editPhCyc2)
              disp(['Executing First Edit phase cycle ' num2str(EP1) ' of ' num2str(length(editPhCyc1)) ', '...
                    ' and Second Edit phase cycle ' num2str(EP2) ' of ' num2str(length(editPhCyc2)) '!!!']);
              outON_epc{EP1}{EP2}=sim_megapress_shapedEdit(npts,sw,Bfield,lw,taus,sys,...
                  editRFon,editTp,editPhCyc1(EP1),editPhCyc2(EP2));
              outOFF_epc{EP1}{EP2}=sim_megapress_shapedEdit(npts,sw,Bfield,lw,taus,sys,...
                  editRFoff,editTp,editPhCyc1(EP1),editPhCyc2(EP2));
              if EP1==1 && EP2==1
                  outON=outON_epc{EP1}{EP2};
                  outOFF=outOFF_epc{EP1}{EP2};
              else
                  outON=op_addScans(outON,outON_epc{EP1}{EP2});
                  outOFF=op_addScans(outOFF,outOFF_epc{EP1}{EP2});
              end
          end
      end

      nu = outOFF.ppm;
      pulse_sequence = 'megapress';
      linewidth = outOFF.linewidth;
      t = outOFF.t;
      omega = outOFF.Bo * gamma * 1e-6;
      m_name = metabolites{ii};

      edit = false;
      fid = outOFF.fids;
      fft = outOFF.specs;
      f_name = sprintf('FIDA_%s_MEGAPRESS_EDITOFF_%.2f_%d_%d_%.2f.mat',m_name, lw, sw, npts, omega);
      save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'edit', 'pulse_sequence');

      edit = true;
      fid = outON.fids;
      fft = outON.specs;
      f_name = sprintf('FIDA_%s_MEGAPRESS_EDITON_%.2f_%d_%d_%.2f.mat',m_name, lw, sw, npts, omega);
      save(fullfile(save_dir, f_name), 'm_name', 'nu', 'fid', 'fft', 'linewidth', 't', 'omega', 'edit', 'pulse_sequence');
    end
end
