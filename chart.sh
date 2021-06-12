#!/bin/bash

rm -f /tmp/chart-parallel.sh *.png

for file in *.raw; do
	name=`basename $file .raw`
	echo ./dma-find.py --name $name --chart >> /tmp/chart-parallel.sh
done

parallel --eta < /tmp/chart-parallel.sh
