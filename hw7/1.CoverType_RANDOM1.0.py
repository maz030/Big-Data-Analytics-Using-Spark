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


# In[8]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)
#inputRDD.first()



# In[21]:

Label=2.0
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')])    .map(lambda V:LabeledPoint(1.0 if V[-1] == Label else 0.0, V[:-1]))### FILLIN ###


# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 10

# In[37]:

(trainingData,testData)=Data.randomSplit([0.7,0.3],seed=255)
#print testData.take(1)



# ### Gradient Boosted Trees
# 
# * Following [this example](http://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts) from the mllib documentation
# 
# * [pyspark.mllib.tree.GradientBoostedTrees documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.GradientBoostedTrees)
# 
# #### Main classes and methods
# 
# * `GradientBoostedTrees` is the class that implements the learning trainClassifier,
#    * It's main method is `trainClassifier(trainingData)` which takes as input a training set and generates an instance of `GradientBoostedTreesModel`
#    * The main parameter from train Classifier are:
#       * **data** – Training dataset: RDD of LabeledPoint. Labels should take values {0, 1}.
#       * categoricalFeaturesInfo – Map storing arity of categorical features. E.g., an entry (n -> k) indicates that feature n is categorical with k categories indexed from 0: {0, 1, ..., k-1}.
#       * **loss** – Loss function used for minimization during gradient boosting. Supported: {“logLoss” (default), “leastSquaresError”, “leastAbsoluteError”}.
#       * **numIterations** – Number of iterations of boosting. (default: 100)
#       * **learningRate** – Learning rate for shrinking the contribution of each estimator. The learning rate should be between in the interval (0, 1]. (default: 0.1)
#       * **maxDepth** – Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 3)
#       * **maxBins** – maximum number of bins used for splitting features (default: 32) DecisionTree requires maxBins >= max categories
#       
#       
# * `GradientBoostedTreesModel` represents the output of the boosting process: a linear combination of classification trees. The methods supported by this class are:
#    * `save(sc, path)` : save the tree to a given filename, sc is the Spark Context.
#    * `load(sc,path)` : The counterpart to save - load classifier from file.
#    * `predict(X)` : predict on a single datapoint (the `.features` field of a `LabeledPont`) or an RDD of datapoints.
#    * `toDebugString()` : print the classifier in a human readable format.

# In[48]:

from time import time
errors={}
for depth in [15]:
    start=time()
    model = RandomForest.trainClassifier(data=trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, impurity="gini", maxDepth=depth)## FILLIN ##
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
        ### FILLIN ###
    print depth,errors[depth]
#print errors

#
## In[43]:
#
#B10 = errors
#
#
## In[44]:
#
## Plot Train/test accuracy vs Depth of trees graph
#get_ipython().magic(u'pylab inline')
#from plot_utils import *
#make_figure([B10],['10Trees'],Title='Boosting using 10% of data')
#
#
## ### Random Forests
## 
## * Following [this example](http://spark.apache.org/docs/latest/mllib-ensembles.html#classification) from the mllib documentation
## 
## * [pyspark.mllib.trees.RandomForest documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.RandomForest)
## 
## **trainClassifier**`(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy='auto', impurity='gini', maxDepth=4, maxBins=32, seed=None)`   
## Method to train a decision tree model for binary or multiclass classification.
## 
## **Parameters:**  
## * *data* – Training dataset: RDD of LabeledPoint. Labels should take values {0, 1, ..., numClasses-1}.  
## * *numClasses* – number of classes for classification.  
## * *categoricalFeaturesInfo* – Map storing arity of categorical features. E.g., an entry (n -> k) indicates that feature n is categorical with k categories indexed from 0: {0, 1, ..., k-1}.  
## * *numTrees* – Number of trees in the random forest.  
## * *featureSubsetStrategy* – Number of features to consider for splits at each node. Supported: “auto” (default), “all”, “sqrt”, “log2”, “onethird”. If “auto” is set, this parameter is set based on numTrees: if numTrees == 1, set to “all”; if numTrees > 1 (forest) set to “sqrt”.
## * *impurity* – Criterion used for information gain calculation. Supported values: “gini” (recommended) or “entropy”.  
## * *maxDepth* – Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 4)  
## * *maxBins* – maximum number of bins used for splitting features (default: 32)
## * *seed* – Random seed for bootstrapping and choosing feature subsets.  
## 
## **Returns:**	
## RandomForestModel that can be used for prediction
#
## In[56]:
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
## In[57]:
#
#RF_10trees = errors
## Plot Train/test accuracy vs Depth of trees graph
#make_figure([RF_10trees],['10Trees'],Title='Random Forests using 10% of data')
#
#
## ### Now plot B10 and RF_10trees performance curves in the same graph
#
## In[61]:
#
#make_figure([B10, RF_10trees],['10Trees', '10Trees'],                                Title='Boosting using 10% of data, Random Forests using 10% of data')
#
#
## In[ ]:



