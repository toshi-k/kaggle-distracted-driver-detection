#!/bin/sh

for i in `seq 1 30`
do
	echo "run with seed: ${i}"
	th main.lua -seed ${i}
	find preprocess/ -type f | grep -E /\([^t]\|.[^e]\) | xargs rm
done
