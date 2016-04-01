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


# Positive results:

confusion matrices for each fold:

    Joy     Desire  Love    Other
    204     28      46      22
    14      232     25      4
    34      27      188     11
    36      36      13      81
    Joy     Desire  Love    Other
    199     15      61      25
    16      213     18      28
    26      17      201     16
    32      24      16      94
    Joy     Desire  Love    Other
    209     20      38      33
    23      219     12      21
    44      27      171     18
    40      25      14      87
    Joy     Desire  Love    Other
    225     18      29      28
    24      215     13      23
    65      12      172     11
    53      15      14      84
    Joy     Desire  Love    Other
    219     30      36      15
    27      213     26      9
    52      14      189     5
    55      27      17      67
    Joy     Desire  Love    Other
    233     20      29      18
    37      201     19      17
    62      11      181     5
    58      16      19      73
    Joy     Desire  Love    Other
    193     12      47      48
    21      205     20      28
    46      15      172     26
    30      23      12      101
    Joy     Desire  Love    Other
    234     15      28      23
    24      201     26      23
    49      11      183     16
    48      16      19      83
    Joy     Desire  Love    Other
    218     13      34      35
    22      222     12      18
    37      22      187     13
    29      33      17      87
    Joy     Desire  Love    Other
    179     16      57      47
    10      206     22      36
    31      20      190     18
    28      16      23      99

# Overall positive results from all folds combined:

Overall confusion matrix:

    Joy     Desire  Love    Other
    2121    161     351     366
    235     2078    153     279
    520     169     1731    175
    433     234     138     855
Averaged confusion matrix:

    Joy     Desire  Love    Other
    211     18      40      29
    21      212     19      20
    44      17      183     13
    40      23      16      85
Overall precision, recall, f1-score and support values for each class:

                Joy         Desire      Love        Other
    precision:  0.64097915  0.78652536  0.72945638  0.51044776
    recall:     0.70723575  0.75701275  0.66705202  0.51506024
    recall:     0.70453289  0.77482814  0.70674042  0.51566265
    f1-score:   0.67247939  0.77148691  0.6968599   0.51274363
    support:    2999        2745        2595        1660
Average precision, recall, f1-score and support values for each class:

                Joy         Desire      Love        Other
    precision:  0.66884396  0.7845121   0.70861263  0.58631526
    recall:     0.70453289  0.77482814  0.70674042  0.51566265
    f1-score:   0.68305063  0.77830682  0.70651473  0.54180708
    support:    299.9       274.5       259.5       166.     

Below are the results when ~1000 negative emotion samples (sadness, anger, disgust, hate) were added to the "other" category of the positive data for balancing,
It is interesting to see that "other" category was improved but some positive emotion labels were negatively affected:

                Joy         Desire      Love        Other
    precision:  0.63089167  0.74927101  0.72471765  0.64202944
    recall:     0.69155184  0.77551958  0.66441788  0.57894737
    f1-score:   0.65639208  0.759734    0.69080176  0.60143182
    support:    299.9       274.5       259.5       266.


# Negative results:

confusion matrices from each fold:

    Sadness Anger   Disgust     Hate    Other
    82      27      17          8       5
    14      61      23          17      6
    10      23      62          18      4
    4       21      18          36      8
    7       8       6           5       12
    Sadness Anger   Disgust     Hate    Other
    104     16      11          5       3
    16      63      15          25      2
    15      13      57          27      4
    11      12      13          49      2
    5       5       7           7       14   
    Sadness Anger   Disgust     Hate    Other
    94      22      19          2       2
    20      65      29          7       0
    11      20      77          6       2
    3       13      41          28      2
    9       4       20          0       5
    Sadness Anger   Disgust     Hate    Other    
    90      30      8           4       7
    18      84      9           5       4
    16      30      49          16      5
    7       17      30          30      3
    6       6       9           2       15
    Sadness Anger   Disgust     Hate    Other
    103     11      11          9       5
    17      54      14          26      9
    11      8       65          23      9
    7       14      17          48      1
    3       4       3           5       23
    Sadness Anger   Disgust     Hate    Other
    95      15      23          4       2
    28      56      15          19      2
    11      14      75          11      5
    6       11      23          46      1
    10      5       12          3       8
    Sadness Anger   Disgust     Hate    Other
    92      22      20          0       5
    12      71      18          12      7
    12      20      73          7       4
    4       20      28          32      2
    8       5       13          1       11
    Sadness Anger   Disgust     Hate    Other
    85      22      13          12      7
    24      61      23          6       6
    13      23      61          17      2
    4       20      18          41      3
    4       3       21          2       8
    Sadness Anger   Disgust     Hate    Other
    96      19      12          4       8
    18      66      29          4       3
    16      23      68          3       6
    11      19      22          31      3
    5       7       6           3       17
    Sadness Anger   Disgust     Hate    Other
    71      48      11          5       4
    9       83      13          15      0
    2       45      56          13      0
    5       20      17          42      2
    3       11      14          3       6
# Overall negative results from all folds combined:

Overall confusion matrix:

    Sadness Anger   Disgust     Hate    Other
    1028    115     155         52      40
    282     567     198         110     46
    168     177     628         129     59
    88      142     228         366     42
    102     51      102         29      95
 
Averaged confusion matrix:

    Sadness Anger   Disgust     Hate    Other
    91      23      14          5       4
    17      66      18          13      3
    11      21      64          14      4
    6       16      22          38      2
    6       5       11          3       11

Overall precision, recall, f1-score and support values for each class:

                Sadness     Anger       Disgust     Hate        Other
    precision:  0.61630695  0.53897338  0.47902365  0.5335277   0.33687943
    recall:     0.73956835  0.4713217   0.54091301  0.42263279  0.25065963
    f1-score:   0.67233486  0.50288248  0.50809061  0.47164948  0.28744327
    support:    1390        1203        1161        866         379

Averaged precision, recall, f1-score and support values for each class:

                Sadness     Anger       Disgust     Hate        Other
    precision:  0.69155112  0.506961      0.495342      0.53947037    0.43788473
    recall:     0.65611511    0.55203168    0.55385352    0.44218123    0.31358464
    f1-score:   0.66992338    0.52175505    0.51874563    0.47337592    0.34974401
    support:    139.          120.3         116.1          86.6          37.9       

Below are the results when ~1000 positive emotion samples (love, desire, joy) were added to the "other" category of negative data for balancing,
It is interesting to see that "other" category was improved but sadness, anger, disgust and hate scores were negatively affected:

                Sadness     Anger       Disgust     Hate        Other
    precision: 0.60313347   0.51409138  0.45666408  0.49632042  0.7127853
    recall:    0.6352518    0.47709366  0.53902151  0.39949211  0.63587221 
    f1-score:  0.61538925   0.49156961  0.48857583  0.41760351  0.66797395   
    support:   139.         120.3       116.1       86.6        137.9      



# Prediction results
Prediction results for the test files can be achieved by running 


    THEANO_FLAGS=mode=FAST_RUN,device=gpuN,floatX=float32  python predict.py


(The results are saved into data/pos_emotions_pred.csv and data/neg_emotions_pred.csv)

# Future work:
* comparison with the other models, e.g. tf-idf with svm might perform better, since the data is not that big
* other deep architectures and hyperparameter search
* ensemble models
