# installation:

sudo  pip install sklearn


sudo  pip install six


sudo  pip install nltk


sudo  pip install wordsegment
 


  # how to install keras: 
    https://github.com/fchollet/keras
    
    
    Dependencies: numpy, scipy, pyyaml, HDF5 and h5py
    
    
    Optional but recommended: cuDNN
    
    
    # install theano:
    
    
    sudo pip install git+git://github.com/Theano/Theano.git
        
        
    sudo  pip install keras
    


# run:
THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python __main__.py 
 
 
where gpuN should be replaced by a real device e.g. gpu0


or alternatively, run on cpu:


python __main__.py

# results:

