#!/bin/bash

docker rmi -f $1
docker build -t $1 .
