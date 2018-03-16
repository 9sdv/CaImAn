#!/bin/bash

docker stop $2
docker rm $2
docker run -p $3:8080 -v $PWD:/CaImAn -v /data:/data -v /data2:/data2 -v /analysis:/analysis -v /scratch:/scratch --name=$2 -t $1
