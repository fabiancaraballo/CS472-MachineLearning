#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd
#
#
import sys
import math
import re
# Node class for the decision tree
import node


train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if p <= 0 or 1-p <= 0:
        return 0
    if p == 1-p:
        return 1
    if p > 0 and 1-p > 0:
        entropy = -p * math.log((p),2) - (1-p)* math.log((1-p),2)
        return entropy


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    if py_pxi == 0 or pxi == 0:
        positive_entropy = 0
    else:
        pos = (py_pxi+0.0)/pxi
        positive_entropy = entropy(pos)
    if py-py_pxi == 0:
        negative_entropy = 0
    else:
        neg = ((py-py_pxi)+0.0)/(total-pxi)
        negative_entropy = entropy(neg)
    
    total_entropy = (py+0.0)/total
    entropy_total = entropy(total_entropy)
    
    infogain = entropy_total - (((pxi+0.0)/total)*positive_entropy) - ((((total-pxi)+0.0)/total)*negative_entropy)
    return infogain





# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    py_pxi = 0
    pxi = 0
    py = 0
    total = len(data)
    for i in data:
        if i[len(data[0])-1] == 1:
            py += 1

    guess = py / (total * 1.0)
    if guess == 1:
        return node.Leaf(varnames, 1)
    elif guess == 0:
        return node.Leaf(varnames, 0)

    if len(varnames) == 1:
        if guess > 0.5 :
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)

    gain = 0;

    for i in range(len(varnames) - 1):
        for j in data:
            if j[i] == 1:
                pxi += 1
            if j[i] == 1 and j[-1] == 1:
                py_pxi += 1
        if infogain(py_pxi, pxi, py, total) > gain :
            gain = infogain(py_pxi, pxi, py, total)
            index = i
        py_pxi = 0
        pxi = 0

    if gain == 0:
        if guess > 0.5:
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)

    leftData = []
    rightData = []

    for i in range(len(data)):
        if data[i][index] == 0:
            list = data[i]
            leftData.append(list)
        else:
            list = data[i]
            rightData.append(list)

    return node.Split(varnames, index, build_tree(leftData, varnames), build_tree(rightData, varnames))
    
    
            


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
# 
def loadAndTrain(trainS,testS,modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)

def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct)/len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print 'Usage: id3.py <train> <test> <model>'
        sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2])

    acc = runTest()
    print "Accuracy: ",acc

if __name__ == "__main__":
    main(sys.argv[1:])
