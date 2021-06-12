#!/bin/bash

rm -f /tmp/parse-parallel.sh

for file in *.raw; do
	name=`basename $file .raw`
	echo "cat $file | ./dma-find.py --name $name --parse" >> /tmp/parse-parallel.sh
done

parallel --eta < /tmp/parse-parallel.sh
