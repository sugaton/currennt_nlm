#!/bin/zsh

dir=$HOME/datas/allwordwsd2007/new/test/data
for file in "d001_" "d002_" "d003_" "d004_" "d005_"
do

fname="validate_"$file
../nnlmwsd --wsdResult result/$fname --val_file $dir/$file --options_file config_val.cfg

done
