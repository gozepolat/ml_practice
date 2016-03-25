import numpy as np
def append_data(path1, path2,  n=1000):
    balanced1=path1[0:path1.rfind(".")] + "_balanced.csv"
    balanced2=path2[0:path2.rfind(".")] + "_balanced.csv"
    balanced_list1=[]
    balanced_list2=[]
    with open(path1, "r") as t1, open(path2, "r") as t2:
        for line1,line2 in zip(path1,path2):
            l1=line1.strip()
            if len(l1)> 6 and l1[-6:] != ",Other":
                pass
                #if np.balanced_list2.append()


"""from sklearn.cross_validation import StratifiedKFold

def load_data():
    # load your data using this function

def create model():
    # create your model using this function

def train_and_evaluate__model(model, data[train], labels[train], data[test], labels[test)):
    model.fit...
    # fit and evaluate here.

if __name__ == "__main__":
    n_folds = 10
    data, labels, header_info = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print ("Cross-validation fold: %d/%d"%(i+1, n_folds))
            model = None # Clearing the NN.
            model = construct_cnn_lstm()
            train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))"""