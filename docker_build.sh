#!/bin/bash

sudo docker rmi -f $1
sudo docker build -t $1 .
