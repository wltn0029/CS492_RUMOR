# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: bottom-up recursive neural networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import os
import Bid_RvNN
import TD_RvNN
import BU_RvNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.testing import assert_array_almost_equal

import time
import datetime
import random
from evaluate import *
#from Util import *

obj = "Twitter15" # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "2" # fold index, choose from 0-4
tag = "_u2b"
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 500
lr = 0.005

treePath = '../resource/data.TD_RvNN.vol_'+str(vocabulary_size)+'.txt' 
#treePath_bu = '../resource/data.BU_RvNN.vol_'+str(vocabulary_size)+'.txt' 

trainPath = "../nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt" 
testPath = "../nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
labelPath = "../resource/"+obj+"_label_All.txt"



#floss = open(lossPath, 'a+')

################################### tools #####################################
def str2matrix(Str, MaxL): # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( MaxL-l ) ]
    wordFreq += ladd 
    wordIndex += ladd 
    #print MaxL, l, len(Str.split(' ')), len(wordFreq)
    #print Str.split(' ')
    return wordFreq, wordIndex 

def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = [1,0,0,0]
       l1 += 1
    if label in labelset_f:
       y_train = [0,1,0,0] 
       l2 += 1
    if label in labelset_t:
       y_train = [0,0,1,0] 
       l3 += 1 
    if label in labelset_u:
       y_train = [0,0,0,1] 
       l4 += 1
    return y_train, l1,l2,l3,l4

def constructTree(tree, tree_type):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        if tree_type == 0:
            node = TD_RvNN.Node_tweet(idx=i)
        else :
            node = BU_RvNN.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j 
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        #print tree[j]['maxL']
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        #nodeC.time = tree[j]['post_t']
        ## not root node ## 
        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        ## root node ##
        else:
           root = nodeC
    ## 3. convert tree to DNN input    
    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    if tree_type ==0:
        x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x) 
    else:
        x_word, x_index, tree = BU_RvNN.gen_nn_inputs(root, only_leaves_have_vals=False) 
    return x_word, x_index, tree, parent_num

################################# loas data ###################################
def loadData():
    print ("loading tree label"),
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()   
    print( len(labelDic))

    print( "reading tree", )## X
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])  
        Vec =  line.split('\t')[5] 
        if treeDic.get(eid) is None:
           treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':Vec}   
    print( 'tree no:', len(treeDic))

    # for i,(eid, data) in enumerate(treeDic.items()):
    #     i += 1
    #     print('--------------------')
    #     print(eid)
    #     print(data[0][1]['parent_num'],len(data[0]))
    #     if i >= 10:
    #         break
    print( "loading train set", )
    tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    for eid in open(trainPath):
        #if c > 8: break
        eid = eid.rstrip()
        if labelDic.get(eid) is None: continue
        if treeDic.get(eid) is None: continue 
        if len(treeDic[eid]) <= 0: 
           #print labelDic[eid]
           continue
        ## 1. load label
        label = labelDic[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        ## 2. construct tree
        #print eid
        x_word_td, x_index_td, tree_td, parent_num = constructTree(treeDic[eid], 0)
        x_word_bu, x_index_bu, tree_bu, _ = constructTree(treeDic[eid], 1)
        tree_train.append([tree_td,tree_bu])
        word_train.append([x_word_td,x_word_bu])
        index_train.append([x_index_td, x_index_bu])
        parent_num_train.append(parent_num)
        #print treeDic[eid]
        #print tree, child_num
        #exit(0)
        c += 1
    print( l1,l2,l3,l4)

    print( "loading test set", )
    tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    for eid in open(testPath):
        #if c > 4: break
        eid = eid.rstrip()
        if labelDic.get(eid) is None: continue
        if treeDic.get(eid) is None: continue
        if len(treeDic[eid]) <= 0: 
            #print labelDic[eid] 
            continue
        ## 1. load label        
        label = labelDic[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        ## 2. construct tree
        x_word_td, x_index_td, tree_td, parent_num = constructTree(treeDic[eid], 0)
        x_word_bu, x_index_bu, tree_bu, _ = constructTree(treeDic[eid], 1)
        tree_test.append([tree_td,tree_bu])
        word_test.append([x_word_td,x_word_bu])
        index_test.append([x_index_td, x_index_bu])
        parent_num_test.append(parent_num)
        c += 1
    print( l1,l2,l3,l4)
    print( "train no:", len(tree_train), len(word_train), len(index_train),len(parent_num_train), len(y_train))
    print( "test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
    print("dim1 for 0:", len(tree_train[0][0]), len(word_train[0][0]), len(index_train[0][0]))
    #print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])
    #print index_train[0]
    #print word_train[0]
    #print tree_train[0]    
    #exit(0)
    return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

##################################### MAIN ####################################        
## 1. load tree & word & index & label
tree_train, word_train, index_train, parent_num_train, y_train, tree_test, \
                word_test, index_test, parent_num_test, y_test = loadData()

## 1.5. Check device and get device (gpu, cpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(device)
## 2. ini RNN model
t0 = time.time()
model = Bid_RvNN.RvNN(vocabulary_size, hidden_dim, Nclass, device=device)
t1 = time.time()
print('Recursive model established,', (t1-t0)/60)


######################

## 3. looping SGD
losses_5, losses = [], []
num_examples_seen = 0
for epoch in range(Nepoch):
    ## one SGD 
    indexs = [i for i in range(len(y_train))]
    random.shuffle(indexs) 
    for i in indexs:
        #print i,
        #print(parent_num_train)

        loss, pred_y = model.forward(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], lr)
        #print loss, pred_y
        losses.append(float(loss))
        num_examples_seen += 1
    print( "epoch=%d: loss=%.4f" % ( epoch, np.mean(losses) ))
    #floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n')
    sys.stdout.flush()
    
    ## cal loss & evaluate
    if epoch % 5 == 0:
       losses_5.append((num_examples_seen, np.mean(losses))) 
       time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
       print( "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
       #floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n') 
       #floss.flush()        
       sys.stdout.flush()
       prediction = []
       for j in range(len(y_test)):
           #print j
           prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]) )   
       res = evaluation_4class(prediction, y_test) 
       print( 'results:', res)
       #floss.write(str(res)+'\n')
       #floss.flush() 
       sys.stdout.flush()
       ## Adjust the learning rate if loss increases
       if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
          lr = lr * 0.5   
          print( "Setting learning rate to %f" % lr)
          #floss.write("Setting learning rate to:"+str(lr)+'\n')
          #floss.flush() 
          sys.stdout.flush()
       #save_model_Recursive_gruEmb(modelPath, model)   
    sys.stdout.flush()
    losses = []
    
#floss.close() 
