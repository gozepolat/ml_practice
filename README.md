# Installation:

    sudo  pip install sklearn


    sudo  pip install six


    sudo  pip install nltk


    sudo  pip install wordsegment
 


  # How to install keras: 
    https://github.com/fchollet/keras
    
    
    Dependencies: numpy, scipy, pyyaml, HDF5 and h5py
    
    
    Optional but recommended: cuDNN
    
    
    # install theano:
    
    
    sudo pip install git+git://github.com/Theano/Theano.git
        
        
    sudo  pip install keras
    


# Running:

    THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python __main__.py 
 
 
    where gpuN should be replaced by a real device e.g. gpu0


    or alternatively, run on cpu:


    python __main__.py

# positive results:

confusion matrix:


    Joy   Desire  Love Other
    209     23      19  49
    15      218     13  29
    63      29      142 26
    62      25      15  164

average precision, recall, f1-score and support values for each class:


                Joy         Desire      Love        Other
    precision:  0.63089167  0.74927101  0.72471765  0.64202944
    recall:     0.69155184  0.77551958  0.66441788  0.57894737
    f1-score:   0.65639208  0.759734    0.69080176  0.60143182
    support:    299.9       274.5       259.5       266.


# negative results:

confusion matrix:


    Sadness Anger   Disgust     Hate    Other
    90      26      12          2       9
    21      66      17          13      3
    12      19      65          14      6
    11      17      18          30      10
    22      6       18          6       86


average precision, recall, f1-score and support values for each class:


                Sadness     Anger       Disgust     Hate        Other
    precision: 0.60313347   0.51409138  0.45666408  0.49632042  0.7127853
    recall:    0.6352518    0.47709366  0.53902151  0.39949211  0.63587221 
    f1-score:  0.61538925   0.49156961  0.48857583  0.41760351  0.66797395   
    support:   139.         120.3       116.1       86.6        137.9      


# future work:
* comparison with the other models, e.g. tf-idf with svm might perform better, since the data is not that big
* other deep architectures and hyperparameter search
* ensemble models