import numpy as np


def exchange_data(path1, path2, n1=1000, n2=1000):
    balanced1 = path1[0:path1.rfind(".")] + "_balanced.csv"
    balanced2 = path2[0:path2.rfind(".")] + "_balanced.csv"
    balanced_list1 = []
    balanced_list2 = []
    ctr1 = 0
    ctr2 = 0
    with open(path1, "r") as t1, open(path2, "r") as t2:
        for line1, line2 in zip(t1, t2):
            l1 = line1.strip()
            l2 = line2.strip()
            if len(l1) > 6 and l1[-6:] != ",Other":
                if np.random.uniform(0, 1) < 0.3 and ctr2 < n2:
                    balanced_list2.append(l1[0:l1.rfind(",")] + ",Other")
                    ctr2 += 1
            if len(l2) > 6 and l2[-6:] != ",Other":
                if np.random.uniform(0, 1) < 0.3 and ctr1 < n1:
                    balanced_list1.append(l2[0:l2.rfind(",")] + ",Other")
                    ctr1 += 1
            balanced_list1.append(l1)
            balanced_list2.append(l2)
    np.random.shuffle(balanced_list1)
    np.random.shuffle(balanced_list2)
    with open(balanced1, "w") as b1:
        for line in balanced_list1:
            b1.write(line+"\n")
    with open(balanced2, "r") as b2:
        for line in balanced_list2:
            b2.write(line+"\n")





