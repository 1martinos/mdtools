#!/bin/sh
for n_epochs in 50 100 150
do
	for d_l in 3 5 10 15
	do
		/home/sam/.pyenv/shims/ipython ./train.py $n_epochs $d_l
	done
done

