#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
import math
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  g = [0.0] * len(data)
  
  for i in range(0, MAX_ITERS):
      gradientb = 0
      gradientW = [0.0] * numvars
      for item in data:
          a = 0
          for n in range(0, numvars):
              a += item[0][n] * w[n]
          a += b
          prediction = sigmoid(a)
          if item[1] == -1:
              g[gradientb] = prediction
          else:
              g[gradientb] = prediction - 1
          gradientb += 1

      for n in range(0, numvars):
          w[n] -= (l2_reg_weight /(len(data)) * w[n])
          for p in range(0, len(data)):
              gr = eta * g[p] * data[p][0][n]
              w[n] -= gr

      for n in range(0, numvars):
          b += eta * g[n]



  return (w,b)

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model
  prob = 0
  for i in range(0, len(model[0])):
      prob += model[0][i] * x[i]
  prob += b

  return sigmoid(prob)
 


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
