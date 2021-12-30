% SimMega2D.m
%
% SPDX-FileCopyrightText: Copyright (C) 2021 S Shermer <lw1660@gmail.com>, Swansea University
% SPDX-License-Identifier: BSD-3-Clause

function [outON,outOFF,outDIFF] = SimMeta2D(sys,p,outfile,save_dir)

% changed output to save to disk instead

[DX,DY] = meshgrid(p.x,p.y);

%Initialize structures:
outON_posxy_epc_rpc  = cell(length(p.x),length(p.y),length(p.editPhCyc1),length(p.editPhCyc2),length(p.refPhCyc1),length(p.refPhCyc2));
outOFF_posxy_epc_rpc = cell(length(p.x),length(p.y),length(p.editPhCyc1),length(p.editPhCyc2),length(p.refPhCyc1),length(p.refPhCyc2));
outON_posxy_epc      = cell(length(p.x),length(p.y),length(p.editPhCyc1),length(p.editPhCyc2));
outOFF_posxy_epc     = cell(length(p.x),length(p.y),length(p.editPhCyc1),length(p.editPhCyc2));
outON_posxy          = cell(length(p.x),length(p.y));
outOFF_posxy         = cell(length(p.x),length(p.y));
outDIFF_posxy        = cell(length(p.x),length(p.y));
outON                = struct([]);
outOFF               = struct([]);
outDIFF              = struct([]);

for X=1:length(p.x)
    for Y=1:length(p.y)
        tic
        for EP1=1:length(p.editPhCyc1)

            for EP2=1:length(p.editPhCyc2)

                for RP1=1:length(p.refPhCyc1)

                    for RP2=1:length(p.refPhCyc2)
                        disp(['Executing X-position ' num2str(X) ' of ' num2str(length(p.x)) ', '...
                            'Y-position ' num2str(Y) ' of ' num2str(length(p.y)) ', '...
                            'First Edit phase cycle ' num2str(EP1) ' of ' num2str(length(p.editPhCyc1)) ', '...
                            'Second Edit phase cycle ' num2str(EP2) ' of ' num2str(length(p.editPhCyc2)) ', '...
                            'First Refoc phase cycle ' num2str(RP1) ' of ' num2str(length(p.refPhCyc1)) ', '...
                            'Second Refoc phase cycle ' num2str(RP2) ' of ' num2str(length(p.refPhCyc2)) '!!!']);

                        outON_posxy_epc_rpc{X,Y,EP1,EP2,RP1,RP2} = sim_megapress_shaped(p.Npts, p.sw,...
                                                                   p.Bfield, p.lw, p.taus, sys,...
                                                                   p.editRFon, p.editTp, ...
                                                                   p.editPhCyc1(EP1), p.editPhCyc2(EP2), ...
                                                                   p.refRF, p.refTp, ...
                                                                   p.Gx, p.Gy, p.x(X), p.y(Y), ...
                                                                   p.refPhCyc1(RP1), p.refPhCyc2(RP2));

                        outOFF_posxy_epc_rpc{X,Y,EP1,EP2,RP1,RP2} = sim_megapress_shaped(p.Npts, p.sw, ...
                                                                   p.Bfield, p.lw, p.taus, sys, ...
                                                                   p.editRFoff, p.editTp, ...
                                                                   p.editPhCyc1(EP1),p.editPhCyc2(EP2), ...
                                                                   p.refRF,p.refTp, ...
                                                                   p.Gx,p.Gy,p.x(X),p.y(Y), ...
                                                                   p.refPhCyc1(RP1),p.refPhCyc2(RP2));

                        if RP1==1 && RP2==1
                            outON_posxy_epc {X,Y,EP1,EP2} = outON_posxy_epc_rpc {X,Y,EP1,EP2,RP1,RP2};
                            outOFF_posxy_epc{X,Y,EP1,EP2} = outOFF_posxy_epc_rpc{X,Y,EP1,EP2,RP1,RP2};
                        else
                            outON_posxy_epc {X,Y,EP1,EP2} = op_addScans(outON_posxy_epc {X,Y,EP1,EP2}, ...
                                                                        outON_posxy_epc_rpc  {X,Y,EP1,EP2,RP1,RP2}, ...
                                                                        xor(RP1==length(p.refPhCyc1), ...
                                                                            RP2==length(p.refPhCyc2)));
                            outOFF_posxy_epc{X,Y,EP1,EP2} = op_addScans(outOFF_posxy_epc{X,Y,EP1,EP2}, ...
                                                                        outOFF_posxy_epc_rpc{X,Y,EP1,EP2,RP1,RP2}, ...
                                                                        xor(RP1==length(p.refPhCyc1), ...
                                                                            RP2==length(p.refPhCyc2)));
                        end

                    end %end of 1st refocusing phase cycle loop

                end %end of 2nd refocusing phase cycle loop.

                if EP1==1 && EP2==1
                    outON_posxy {X,Y} = outON_posxy_epc {X,Y,EP1,EP2};
                    outOFF_posxy{X,Y} = outOFF_posxy_epc{X,Y,EP1,EP2};
                else
                    outON_posxy{X,Y}  = op_addScans(outON_posxy {X,Y}, outON_posxy_epc {X,Y,EP1,EP2});
                    outOFF_posxy{X,Y} = op_addScans(outOFF_posxy{X,Y}, outOFF_posxy_epc{X,Y,EP1,EP2});
                end

                outDIFF_posxy{X,Y}=op_subtractScans(outON_posxy{X,Y}, outOFF_posxy{X,Y});

            end %end of 1st editing phase cycle loop.

        end %end of 2nd editing phase cycle loop.

        outON  = op_addScans(outON, outON_posxy{X,Y});
        outOFF = op_addScans(outOFF,outOFF_posxy{X,Y});
        toc

    end %end of spatial loop (parfor) in y direction.

end %end of spatial loop (parfor) in x direction.

outDIFF = op_subtractScans(outON,outOFF);

% collect output
t   = outOFF.t;
ppm = outOFF.ppm;

fids_ON_posxy         = cellfun(@(x)x.fids,    outON_posxy,'UniformOutput',false);
fids_ON_posxy_epc     = cellfun(@(x)x.fids,    outON_posxy_epc,'UniformOutput',false);
fids_ON_posxy_epc_rpc = cellfun(@(x)x.fids,    outON_posxy_epc_rpc,'UniformOutput',false);

fids_OFF_posxy         = cellfun(@(x)x.fids,   outOFF_posxy,'UniformOutput',false);
fids_OFF_posxy_epc     = cellfun(@(x)x.fids,   outOFF_posxy_epc,'UniformOutput',false);
fids_OFF_posxy_epc_rpc = cellfun(@(x)x.fids,   outOFF_posxy_epc_rpc,'UniformOutput',false);

specs_ON_posxy         = cellfun(@(x)x.specs,  outON_posxy,'UniformOutput',false);
specs_ON_posxy_epc     = cellfun(@(x)x.specs,  outON_posxy_epc,'UniformOutput',false);
specs_ON_posxy_epc_rpc = cellfun(@(x)x.specs,  outON_posxy_epc_rpc,'UniformOutput',false);

specs_OFF_posxy         = cellfun(@(x)x.specs, outOFF_posxy,'UniformOutput',false);
specs_OFF_posxy_epc     = cellfun(@(x)x.specs, outOFF_posxy_epc,'UniformOutput',false);
specs_OFF_posxy_epc_rpc = cellfun(@(x)x.specs, outOFF_posxy_epc_rpc,'UniformOutput',false);

if ~exist('save_dir','var')
  save_dir = '.';
end

if ~exist('outfile','var')
    outfile = 'output';
end

save(fullfile(save_dir,outfile),'sys','p','t','ppm','fids*','specs*')

end % function
