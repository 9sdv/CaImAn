#!/bin/bash

sudo docker stop $2
sudo docker rm $2
sudo docker run -p $3:8080 -v /home/jack/code/CaImAn:/CaImAn -v /data:/data -v /data2:/data2 -v /analysis:/analysis -v /scratch:/scratch --name=$2 -t $1
