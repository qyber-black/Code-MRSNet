# Train best CNN model (sum)
./mrsnet.py train -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/100000-1 -e 100 -k 5 --norm sum --acquisitions edit_off edit_on --datatype real -m cnn_medium_sigmoid_pool -b 16 -v

# Benchmark best CNN model (sum)
./mrsnet.py benchmark  --model ./data/model/cnn_medium_sigmoid_pool/Cr-GABA-Gln-Glu-NAA/megapress/edit_off-edit_on/real/sum/16/100/fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_100000-1/KFold_5-1/fold-0 --norm max -v

# Train best CNN model (max!) To check if this may actually be better than sum!
./mrsnet.py train -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/100000-1 -e 100 -k 5 --norm max --acquisitions edit_off edit_on --datatype real -m cnn_medium_sigmoid_pool -b 16 -v

# Benchmark best CNN model (max)
./mrsnet.py benchmark  --model ./data/model/cnn_medium_sigmoid_pool/Cr-GABA-Gln-Glu-NAA/megapress/edit_off-edit_on/real/max/16/100/fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_100000-1/KFold_5-1/fold-0 --norm max -v

# Model selection for CNN
MRSNET_DEV=selectgpo_optimise_noload ./mrsnet.py select -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1 -e 100 -k 5 --method gpo ./data/model-cnn/selection-spec/cnn-simple-all.json -v -r 150

# Model selection for CNN with extended CNN parameters
n=0; while [ $n -lt 100 ]; do MRSNET_DEV=selectgpo_optimise_noload ./mrsnet.py select -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1 -e 100 -k 5 --method gpo ./data/model-cnn/selection-spec/cnn-para-all.json -v -r 250; n="`expr $n + 1`"; done

# Model selection on CNN para search (without actual search)
MRSNET_DEV=selectgpo_no_search ./mrsnet.py select -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1 -e 100 -k 5 --method gpo ./data/model-cnn/selection-spec/cnn-para-all.json -v -r 150

# Model selection for cnn with reduce para search space
./mrsnet.py select -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1 -e 40 -k 0.8 --method gpo ./data/model-cnn/selection-spec/cnn-para-reduced.json -v -r 500

# sim2real example
python3 mrsnet.py sim2real --source fid-a-2d --linewidth 2.0 --noise_mc_trials 100 --noise_sigma 0.03 -v