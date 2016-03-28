installation:

sudo -H pip install sklearn
sudo -H pip install six
sudo -H pip install nltk
sudo -H pip install wordsegment
# sudo -H pip install pandas

# keras: https://github.com/fchollet/keras
# Dependencies
numpy, scipy, pyyaml, HDF5 and h5py
# Optional but recommended: 
cuDNN
# install theano
sudo -H pip install git+git://github.com/Theano/Theano.git
# install keras
sudo -H pip install keras
    


run:
THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python __main__.py  # where gpuN should be replaced by a real device e.g. gpu0
or alternatively, run on cpu:
python __main__.py