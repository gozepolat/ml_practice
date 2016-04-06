# Negative model training results with additional ~600 samples from positive data (joy, desire, love categories) as "other":

    overall confusion matrix:
    Sadness, Anger, Disgust, Hate, Other
    [[914 152 154  45 125]
     [223 584 223 116  57]
     [120 173 699 123  46]
     [ 74 154 244 327  67]
     [130  61 108  45 635]]
    overall precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    ('precision', array([ 0.6255989 ,  0.51957295,  0.4894958 ,  0.49847561,  0.6827957 ]))
    ('recall', array([ 0.65755396,  0.48545303,  0.60206718,  0.37759815,  0.64862104]))
    ('fscore', array([ 0.64117853,  0.50193382,  0.53997683,  0.42969777,  0.66526977]))
    ('support', array([1390, 1203, 1161,  866,  979]))
    end of cross-validation for negative task
    total confusion matrix
    [[91 15 15  4 12]
     [22 58 22 11  5]
     [12 17 69 12  4]
     [ 7 15 24 32  6]
     [13  6 10  4 63]]
    average confusion matrix
    [[91 15 15  4 12]
     [22 58 22 11  5]
     [12 17 69 12  4]
     [ 7 15 24 32  6]
     [13  6 10  4 63]]
    average precision, recall, fscore and support values for each class:
    Sadness, Anger, Disgust, Hate, Other
    [[   0.62782706    0.52834107    0.49866678    0.51577579    0.69179872]
     [   0.65755396    0.48545455    0.60218833    0.37722534    0.64854829]
     [   0.64152894    0.49725961    0.53855026    0.42239555    0.66580204]
     [ 139.          120.3         116.1          86.6          97.9       ]]

# Positive model training results with additional ~600 samples from negative data (Sadness, Anger, Disgust, Hate) as "other":

    overall confusion matrix:
    Joy, Desire, Love, Other
    [[2041  214  402  342]
     [ 196 2142  177  230]
     [ 434  158 1812  191]
     [ 434  293  186 1347]]
    overall scores
    overall precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    ('precision', array([ 0.65732689,  0.76309227,  0.70314319,  0.63838863]))
    ('recall', array([ 0.68056019,  0.78032787,  0.6982659 ,  0.5960177 ]))
    ('fscore', array([ 0.66874181,  0.77161383,  0.70069606,  0.61647597]))
    ('support', array([2999, 2745, 2595, 2260]))
    end of cross-validation for positive task
    total confusion matrix
    [[2041  214  402  342]
     [ 196 2142  177  230]
     [ 434  158 1812  191]
     [ 434  293  186 1347]]
    average confusion matrix
    [[204  21  40  34]
     [ 19 214  17  23]
     [ 43  15 181  19]
     [ 43  29  18 134]]
    average precision, recall, fscore and support values for each class:
    Joy, Desire, Love, Other
    [[   0.66193922    0.76618227    0.71075329    0.64688828]
     [   0.68055964    0.78028666    0.69820612    0.5960177 ]
     [   0.66828425    0.77162335    0.70088388    0.6161919 ]
     [ 299.9         274.5         259.5         226.        ]]
