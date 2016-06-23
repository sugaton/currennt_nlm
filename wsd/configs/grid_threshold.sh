#!/bin/zsh

for th in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do

echo $th
../nnlmwsd --wsdResult result/threshold_$th --wsd_threshold $th --options_file config2.cfg > tmpout
tail -n 1 tmpout

done
