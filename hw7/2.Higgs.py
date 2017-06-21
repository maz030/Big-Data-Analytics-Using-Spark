# -*- coding: utf-8 -*-
#Name: Mayu Zhang
# Email: maz030@eng.ucsd.edu
# PID: A53205212
from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8



# In[2]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# ### Higgs data set
# * **URL:** http://archive.ics.uci.edu/ml/datasets/HIGGS#  
# * **Abstract:** This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
# 
# **Data Set Information:**  
# The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.
# 
# 

# In[5]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)
#inputRDD.take(5)


# In[8]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')])     .map(lambda x: LabeledPoint(x[0], x[1:]))

# In[9]:

Data1=Data.sample(False,0.1,seed=255).cache()
(trainingData,testData)=Data1.randomSplit([0.7,0.3],seed=255)


# In[11]:

from time import time
errors={}
for depth in [10]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(data=trainingData, categoricalFeaturesInfo={}, numIterations=10, learningRate = 0.25, maxDepth=depth)##FILLIN to generate 10 trees ##
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted) ### FILLIN ###
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth]
#print errors

#
## In[12]:
#
#B10 = errors
#
#
## In[13]:
#
## Plot Train/test accuracy vs Depth of trees graph
#get_ipython().magic(u'pylab inline')
#from plot_utils import *
#make_figure([B10],['10Trees'],Title='Boosting using 10% of data')
#
#
## In[15]:
#
#from time import time
#errors={}
#for depth in [1,3,6,10,15,20]:
#    start=time()
#    model = RandomForest.trainClassifier(data=trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, impurity="gini", maxDepth=depth)## FILLIN ##
#    #print model.toDebugString()
#    errors[depth]={}
#    dataSets={'train':trainingData,'test':testData}
#    for name in dataSets.keys():  # Calculate errors on train and test sets
#        data=dataSets[name]
#        Predicted=model.predict(data.map(lambda x: x.features))
#        LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted)
#        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
#        errors[depth][name]=Err
#        ### FILLIN ###
#    print depth,errors[depth],int(time()-start),'seconds'
#print errors
#
#
## In[16]:
#
#RF_10trees = errors
## Plot Train/test accuracy vs Depth of trees graph
#make_figure([RF_10trees],['10Trees'],Title='Random Forests using 10% of data')
#
#
## In[17]:
#
#make_figure([B10, RF_10trees],['10Trees', '10Trees'],                                Title='Boosting using 10% of data, Random Forests using 10% of data')
#
#
## In[ ]:



