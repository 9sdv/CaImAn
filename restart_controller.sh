# this script restarts the ipython controller for ipyparallel. run this if there
# is a hub connection timeout. the argument is the docker container's name

sudo docker exec $1 ipcontroller --ip="*"
