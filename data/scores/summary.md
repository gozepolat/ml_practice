
# Positive results:

Overall confusion matrix:

    Joy    Desire   Love    Other
    2137    225     354     283
    204     2193    155     193
    513     223     1731    128
    432     302     141     785
Overall precision, recall, fscore and support values for each class:

                Joy            Desire         Love           Other
    precision:  0.65033475     0.745158       0.72700546     0.56515479
    recall:     0.71257086     0.7989071      0.66705202     0.47289157
    f1-score:   0.68003182     0.77109705     0.69573955     0.51492293
    support:    2999           2745           2595           1660
    
Average confusion matrix:

    Joy    Desire   Love    Other
    213     22      35      28
    20      219     15      19
    51      22      173     12
    43      30      14      78
Average precision, recall, fscore and support values for each class:

                Joy           Desire        Love          Other
    precision:  0.65486451    0.75007112    0.7310487     0.57161747
    recall:     0.71259755    0.79888388    0.66698396    0.47289157
    f1-score:   0.67966667    0.77158229    0.69443533    0.51389058
    support:    299.9         274.5         259.5         166.        

Below are the results when ~1000 negative emotion samples (sadness    anger    disgust    hate) were added to the "other" category of negative data for balancing,
It is interesting to see that "other" category was improved but some positive emotion labels were negatively affected:

                Joy         Desire      Love        Other
    precision:  0.63089167  0.74927101  0.72471765  0.64202944
    recall:     0.69155184  0.77551958  0.66441788  0.57894737
    f1-score:   0.65639208  0.759734    0.69080176  0.60143182
    support:    299.9       274.5       259.5       266.


# Negative results:

overall confusion matrix:

    Sadness Anger   Disgust Hate    Other
    956     201     140     66      27
    234     642     156     139     32
    150     213     573     190     35
    87      172     201     377     29
    108     68      93      42      68
Overall precision, recall, fscore and support values for each class:

                Sadness     Anger       Disgust     Hate        Other
    precision: 0.6228013    0.49537037  0.49269132  0.46314496  0.35602094
    recall:    0.68776978   0.53366584  0.49354005  0.43533487  0.17941953
    f1-score:  0.65367521   0.51380552  0.49311532  0.44880952  0.23859649
    support:   1390         1203        1161        866         379

Average confusion matrix:

    Sadness Anger   Disgust Hate    Other
    95      20      14      6       2
    23      64      15      13      3
    15      21      57      19      3
    8       17      20      37      2
    10      6       9       4       6
Average precision, recall, fscore and support values for each class:

                Sadness     Anger       Disgust     Hate        Other
    precision: 0.62701225    0.50563334    0.49946575    0.49500908    0.37517394
    recall:    0.68776978    0.53364325    0.49341291    0.43548516    0.17944523
    f1-score:  0.6522712     0.51405447    0.49050092    0.45073579    0.23422866
    support:   139.          120.3         116.1          86.6          37.9

Below are the results when ~1000 positive emotion samples (love    desire    joy) were added to the "other" category of negative data for balancing,
It is interesting to see that "other" category was improved but sadness    anger    disgust and hate scores were negatively affected:

                Sadness     Anger       Disgust     Hate        Other
    precision: 0.60313347   0.51409138  0.45666408  0.49632042  0.7127853
    recall:    0.6352518    0.47709366  0.53902151  0.39949211  0.63587221 
    f1-score:  0.61538925   0.49156961  0.48857583  0.41760351  0.66797395   
    support:   139.         120.3       116.1       86.6        137.9      
