#!/bin/bash

IMG=ml9951/ssd
cd ..

RM=--rm

nvidia-docker run -it $RM \
	-w /workspace/ssd.pytorch \
	-e DB_PASSWORD=password \
	-e DB_URL=$(ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d: | head -n1)\
	--ipc=host \
	-v $(pwd):/workspace \
	$IMG /bin/bash
