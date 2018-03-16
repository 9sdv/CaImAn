FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y git wget
RUN apt-get install -y bzip2
RUN apt-get install -y gcc
RUN apt-get install -y libgtk2.0-0
RUN apt-get install -y vim
RUN apt-get install -y libgeos-dev
RUN apt-get install -y python2.7
RUN apt-get -y install python-pip
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install Cython
RUN pip install opencv-python
RUN pip install --upgrade numpy
RUN apt-get install -y python-tk
RUN pip install h5py
RUN pip install IPython
RUN pip install ipyparallel
RUN pip install psutil
RUN pip install tifffile
RUN pip install Bokeh
RUN pip install cvxpy
RUN pip install Pillow
RUN pip install descartes
ADD . /CaImAn
WORKDIR /CaImAn/
RUN pip install -r requirements_pip.txt
RUN apt-get update
RUN apt-get install -y libc6-i386
RUN apt-get install -y libsm6 libxrender1
RUN python setup.py build_ext --inplace
RUN python setup.py develop
RUN pip install git+git://github.com/9sdv/sima.git@wrapper_sequences
WORKDIR /CaImAn

ENV OMP_NUM_THREADS 1
CMD ipcontroller --ip="*" &

RUN apt-get install -y ipython-notebook
RUN apt-get install -y ffmpeg
RUN pip install jupyter
RUN pip intall jupyter_core --upgrade
CMD jupyter notebook --ip 0.0.0.0 --allow-root --port=8080
EXPOSE 8080
