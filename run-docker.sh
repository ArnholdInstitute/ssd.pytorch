#!/bin/bash

IMG=ml9951/ssd2
cd ..

RM=--rm

if [ -z $AWS_SECRET_ACCESS_KEY ]; then
	eval $(awk -F ' = ' 'NF == 2 {print toupper($1)"="$2 }' < ~/.aws/credentials)
fi

nvidia-docker run -it $RM \
	-w /workspace/ssd.pytorch \
	-e "AWS_DEFAULT_REGION=us-west-2" \
	-e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" \
	-e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" \
	--ipc=host \
	-v $(pwd):/workspace \
	$IMG /bin/bash
