

# Train best CNN model
 ./mrsnet.py train -d ./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/100000-1 -e 100 -k 5 --norm sum --acquisitions edit_off edit_on --datatype real -m cnn_medium_sigmoid_pool -b 16 -v

