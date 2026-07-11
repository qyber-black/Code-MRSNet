% SimPress2D.m
%
% SPDX-FileCopyrightText: Dana Goerzen and Jamie Near, McGill University 2021.
% SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
% SPDX-License-Identifier: BSD-3-Clause
%
% 2D spatially-resolved PRESS simulation wrapper, analogous to SimMega2D but for
% an unedited (single sub-spectrum) short-TE PRESS experiment. Coherence-order
% filtering is used (sim_press_shaped) instead of refocusing-pulse phase cycling,
% so there is no edit/phase-cycle dimension. The simulation is run over a grid of
% spatial positions inside the voxel and summed, then amplitude-scaled for the
% number of positions and the simulated-region/voxel ratio (as in FID-A
% run_simPressShaped). The summed spectrum is returned and the full result is
% saved to disk.

function out = SimPress2D(sys,p,outfile,save_dir)

out_posxy = cell(length(p.x),length(p.y));
out       = struct([]);

for X=1:length(p.x)
    for Y=1:length(p.y)
        tic
        disp(['Executing X-position ' num2str(X) ' of ' num2str(length(p.x)) ', '...
              'Y-position ' num2str(Y) ' of ' num2str(length(p.y)) '!!!']);
        out_posxy{X,Y} = sim_press_shaped(p.Npts, p.sw, p.Bfield, p.lw, sys, ...
                                          p.tau1, p.tau2, ...
                                          p.refRF, p.refTp, p.x(X), p.y(Y), ...
                                          p.Gx, p.Gy, p.flipAngle, p.centreFreq);
        out = op_addScans(out, out_posxy{X,Y});
        toc
    end %end of spatial loop in y direction.
end %end of spatial loop in x direction.

% For consistent scaling across different shaped simulations:
% 1. Scale down by the total number of simulations run (they were all added).
numSims = length(p.x)*length(p.y);
out = op_ampScale(out, 1/numSims);
% 2. Scale by the simulated-region size relative to the voxel size.
fovX = max(p.x)-min(p.x);
fovY = max(p.y)-min(p.y);
if fovX > 0 && fovY > 0
  voxRatio = (p.thkX*p.thkY)/(fovX*fovY);
  out = op_ampScale(out, 1/voxRatio);
end

% collect output
t   = out.t;
ppm = out.ppm;
fids  = out.fids;
specs = out.specs;

if ~exist('save_dir','var')
  save_dir = '.';
end
if ~exist('outfile','var')
  outfile = 'output';
end

save(fullfile(save_dir,outfile),'sys','p','t','ppm','fids','specs')

end % function
