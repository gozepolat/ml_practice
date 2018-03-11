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

    THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python main.py  
 
    where gpuN should be replaced by a real device e.g. gpu0

    or alternatively, run on cpu:

    python main.py

# Prediction results
Prediction results for the test files can be acquired by running 


    THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python predict.py


(The results are saved into data/pos_emotions_pred.csv and data/neg_emotions_pred.csv)

# Overall and detailed results:

    scores/summary.md
    scores/all_results_on_a_single_machine.md
# Current best scores are available at:

    scores/current_best_scores.md

# Future work:
* Comparison with the other models, e.g. tf-idf with svm might perform better, since the data is not that big
* Other deep architectures and hyperparameter search
* Ensemble models
