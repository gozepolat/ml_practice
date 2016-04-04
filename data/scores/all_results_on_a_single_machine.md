# Overall results for the negative model:

    overall confusion matrix:
    Joy, Desire, Love, Other
    [[2137  225  354  283]
     [ 204 2193  155  193]
     [ 513  223 1731  128]
     [ 432  302  141  785]]
    overall scores
    overall precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.65033475,  0.745158  ,  0.72700546,  0.56515479]))
    ('recall', array([ 0.71257086,  0.7989071 ,  0.66705202,  0.47289157]))
    ('fscore', array([ 0.68003182,  0.77109705,  0.69573955,  0.51492293]))
    ('support', array([2999, 2745, 2595, 1660]))
    end of cross-validation for positive task
    average confusion matrix
    [[213  22  35  28]
     [ 20 219  15  19]
     [ 51  22 173  12]
     [ 43  30  14  78]]
    average precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    [[   0.65486451    0.75007112    0.7310487     0.57161747]
     [   0.71259755    0.79888388    0.66698396    0.47289157]
     [   0.67966667    0.77158229    0.69443533    0.51389058]
     [ 299.9         274.5         259.5         166.        ]]

# Overall results for the negative model:

    overall confusion matrix:
    Sadness, Anger, Disgust, Hate, Other
    [[956 201 140  66  27]
     [234 642 156 139  32]
     [150 213 573 190  35]
     [ 87 172 201 377  29]
     [108  68  93  42  68]]
    overall precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.6228013 ,  0.49537037,  0.49269132,  0.46314496,  0.35602094]))
    ('recall', array([ 0.68776978,  0.53366584,  0.49354005,  0.43533487,  0.17941953]))
    ('fscore', array([ 0.65367521,  0.51380552,  0.49311532,  0.44880952,  0.23859649]))
    ('support', array([1390, 1203, 1161,  866,  379]))
    end of cross-validation for negative task
    average confusion matrix
    [[95 20 14  6  2]
     [23 64 15 13  3]
     [15 21 57 19  3]
     [ 8 17 20 37  2]
     [10  6  9  4  6]]
    average precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    [[   0.62701225    0.50563334    0.49946575    0.49500908    0.37517394]
     [   0.68776978    0.53364325    0.49341291    0.43548516    0.17944523]
     [   0.6522712     0.51405447    0.49050092    0.45073579    0.23422866]
     [ 139.          120.3         116.1          86.6          37.9       ]]

# Results from each fold for the negative model:

    Cross-validation fold: 1/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[99 17 19  3  1]
     [19 59 28 11  4]
     [14 20 75  8  0]
     [ 8 14 32 30  3]
     [21  4  5  3  5]]

    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.61490683,  0.51754386,  0.47169811,  0.54545455,  0.38461538]))
    ('recall', array([ 0.71223022,  0.48760331,  0.64102564,  0.34482759,  0.13157895]))
    ('fscore', array([ 0.66      ,  0.50212766,  0.54347826,  0.42253521,  0.19607843]))
    ('support', array([139, 121, 117,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 2/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[109  17   6   3   4]
     [ 25  72  13  10   1]
     [ 26  21  50  15   4]
     [ 18  21  12  34   2]
     [  9   9   7   3  10]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.5828877 ,  0.51428571,  0.56818182,  0.52307692,  0.47619048]))
    ('recall', array([ 0.78417266,  0.59504132,  0.43103448,  0.3908046 ,  0.26315789]))
    ('fscore', array([ 0.66871166,  0.55172414,  0.49019608,  0.44736842,  0.33898305]))
    ('support', array([139, 121, 116,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 3/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[106  18  10   2   3]
     [ 26  66  16   6   7]
     [ 20  23  59   9   5]
     [  7  12  24  36   8]
     [ 12   5  11   2   8]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.61988304,  0.53225806,  0.49166667,  0.65454545,  0.25806452]))
    ('recall', array([ 0.76258993,  0.54545455,  0.50862069,  0.4137931 ,  0.21052632]))
    ('fscore', array([ 0.68387097,  0.53877551,  0.5       ,  0.50704225,  0.23188406]))
    ('support', array([139, 121, 116,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 4/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[78 33 13 10  5]
     [22 71 16  8  3]
     [ 8 23 57 27  1]
     [ 6 10 24 45  2]
     [ 6 11 14  3  4]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.65      ,  0.47972973,  0.45967742,  0.48387097,  0.26666667]))
    ('recall', array([ 0.56115108,  0.59166667,  0.49137931,  0.51724138,  0.10526316]))
    ('fscore', array([ 0.6023166 ,  0.52985075,  0.475     ,  0.5       ,  0.1509434 ]))
    ('support', array([139, 120, 116,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 5/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[108   8  18   4   1]
     [ 28  53  22  15   2]
     [ 19  12  67  17   1]
     [ 11  15  26  34   1]
     [ 18   3   9   5   3]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.58695652,  0.58241758,  0.47183099,  0.45333333,  0.375     ]))
    ('recall', array([ 0.77697842,  0.44166667,  0.57758621,  0.3908046 ,  0.07894737]))
    ('fscore', array([ 0.66873065,  0.50236967,  0.51937984,  0.41975309,  0.13043478]))
    ('support', array([139, 120, 116,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 6/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[100  12  20   4   3]
     [ 31  63   7  17   2]
     [ 13  17  57  21   8]
     [  9  14  23  37   4]
     [  9  10   8   4   7]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.61728395,  0.54310345,  0.49565217,  0.44578313,  0.29166667]))
    ('recall', array([ 0.71942446,  0.525     ,  0.49137931,  0.42528736,  0.18421053]))
    ('fscore', array([ 0.66445183,  0.53389831,  0.49350649,  0.43529412,  0.22580645]))
    ('support', array([139, 120, 116,  87,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 7/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[97 15 24  1  2]
     [19 67 18  8  8]
     [13 18 66 13  6]
     [ 4 19 24 35  4]
     [ 8  8 11  0 11]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.68794326,  0.52755906,  0.46153846,  0.61403509,  0.35483871]))
    ('recall', array([ 0.69784173,  0.55833333,  0.56896552,  0.40697674,  0.28947368]))
    ('fscore', array([ 0.69285714,  0.54251012,  0.50965251,  0.48951049,  0.31884058]))
    ('support', array([139, 120, 116,  86,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 8/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[92 23 12  9  3]
     [33 65 11  9  2]
     [16 29 48 20  3]
     [ 8 29 12 35  2]
     [ 7  6 14  5  6]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.58974359,  0.42763158,  0.49484536,  0.44871795,  0.375     ]))
    ('recall', array([ 0.6618705 ,  0.54166667,  0.4137931 ,  0.40697674,  0.15789474]))
    ('fscore', array([ 0.62372881,  0.47794118,  0.45070423,  0.42682927,  0.22222222]))
    ('support', array([139, 120, 116,  86,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 9/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[84 38  9  7  1]
     [14 74 18 14  0]
     [14 36 50 14  2]
     [ 8 28 14 35  1]
     [ 9 11  6  5  7]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.65116279,  0.39572193,  0.51546392,  0.46666667,  0.63636364]))
    ('recall', array([ 0.60431655,  0.61666667,  0.43103448,  0.40697674,  0.18421053]))
    ('fscore', array([ 0.62686567,  0.48208469,  0.46948357,  0.43478261,  0.28571429]))
    ('support', array([139, 120, 116,  86,  38]))
    Sadness, Anger, Disgust, Hate, Other
    Cross-validation fold: 10/10
    Sadness, Anger, Disgust, Hate, Other
    confusion matrix:
    [[83 20  9 23  4]
     [17 52  7 41  3]
     [ 7 14 44 46  5]
     [ 8 10 10 56  2]
     [ 9  1  8 12  7]]
    precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.66935484,  0.53608247,  0.56410256,  0.31460674,  0.33333333]))
    ('recall', array([ 0.5971223 ,  0.43333333,  0.37931034,  0.65116279,  0.18918919]))
    ('fscore', array([ 0.63117871,  0.47926267,  0.45360825,  0.42424242,  0.24137931]))
    ('support', array([139, 120, 116,  86,  37]))
    Sadness, Anger, Disgust, Hate, Other

# Results from each fold for the positive model:

    Cross-validation fold: 1/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[192  20  51  37]
     [ 17 216  16  26]
     [ 34  18 185  23]
     [ 37  24  13  92]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.68571429,  0.77697842,  0.69811321,  0.51685393]))
    ('recall', array([ 0.64      ,  0.78545455,  0.71153846,  0.55421687]))
    ('fscore', array([ 0.66206897,  0.78119349,  0.7047619 ,  0.53488372]))
    ('support', array([300, 275, 260, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 2/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[192  29  52  27]
     [ 11 238  14  12]
     [ 29  25 195  11]
     [ 32  47  11  76]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.72727273,  0.7020649 ,  0.71691176,  0.6031746 ]))
    ('recall', array([ 0.64      ,  0.86545455,  0.75      ,  0.45783133]))
    ('fscore', array([ 0.68085106,  0.7752443 ,  0.73308271,  0.52054795]))
    ('support', array([300, 275, 260, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 3/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[209  26  45  20]
     [ 25 229  12   9]
     [ 51  42 161   6]
     [ 44  46  13  63]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.63525836,  0.66763848,  0.6969697 ,  0.64285714]))
    ('recall', array([ 0.69666667,  0.83272727,  0.61923077,  0.37951807]))
    ('fscore', array([ 0.6645469 ,  0.74110032,  0.65580448,  0.47727273]))
    ('support', array([300, 275, 260, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 4/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[228  16  29  27]
     [ 21 214  13  27]
     [ 52  15 183  10]
     [ 60  18  15  73]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.63157895,  0.81368821,  0.7625    ,  0.53284672]))
    ('recall', array([ 0.76      ,  0.77818182,  0.70384615,  0.43975904]))
    ('fscore', array([ 0.68986384,  0.79553903,  0.732     ,  0.48184818]))
    ('support', array([300, 275, 260, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 5/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[186  34  45  35]
     [ 13 219  27  16]
     [ 41  19 189  11]
     [ 36  32  17  81]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.67391304,  0.72039474,  0.67985612,  0.56643357]))
    ('recall', array([ 0.62      ,  0.79636364,  0.72692308,  0.48795181]))
    ('fscore', array([ 0.64583333,  0.75647668,  0.70260223,  0.52427184]))
    ('support', array([300, 275, 260, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 6/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[220  18  32  30]
     [ 36 203  17  18]
     [ 67  16 170   6]
     [ 47  16  21  82]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.59459459,  0.80237154,  0.70833333,  0.60294118]))
    ('recall', array([ 0.73333333,  0.74087591,  0.65637066,  0.4939759 ]))
    ('fscore', array([ 0.65671642,  0.77039848,  0.68136273,  0.54304636]))
    ('support', array([300, 274, 259, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 7/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[227  19  28  26]
     [ 23 223  12  16]
     [ 69  26 152  12]
     [ 36  39   6  85]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.63943662,  0.72638436,  0.76767677,  0.61151079]))
    ('recall', array([ 0.75666667,  0.81386861,  0.58687259,  0.51204819]))
    ('fscore', array([ 0.69312977,  0.767642  ,  0.66520788,  0.55737705]))
    ('support', array([300, 274, 259, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 8/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[224  32  26  18]
     [ 15 229  18  12]
     [ 51  22 172  14]
     [ 53  30  13  70]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.65306122,  0.73162939,  0.7510917 ,  0.61403509]))
    ('recall', array([ 0.74666667,  0.83576642,  0.66409266,  0.42168675]))
    ('fscore', array([ 0.69673406,  0.7802385 ,  0.70491803,  0.5       ]))
    ('support', array([300, 274, 259, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 9/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[222   9  30  39]
     [ 19 207  16  32]
     [ 41  19 183  16]
     [ 33  18  24  91]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.7047619 ,  0.81818182,  0.72332016,  0.51123596]))
    ('recall', array([ 0.74      ,  0.75547445,  0.70656371,  0.54819277]))
    ('fscore', array([ 0.72195122,  0.78557875,  0.71484375,  0.52906977]))
    ('support', array([300, 274, 259, 166]))
    Joy, Desire, Love, Other
    Cross-validation fold: 10/10
    Joy, Desire, Love, Other
    confusion matrix:
    [[237  22  16  24]
     [ 24 215  10  25]
     [ 78  21 141  19]
     [ 54  32   8  72]]
    precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.60305344,  0.74137931,  0.80571429,  0.51428571]))
    ('recall', array([ 0.79264214,  0.78467153,  0.54440154,  0.43373494]))
    ('fscore', array([ 0.6849711 ,  0.76241135,  0.64976959,  0.47058824]))
    ('support', array([299, 274, 259, 166]))
    Joy, Desire, Love, Other