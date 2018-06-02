#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest

from sklearn.model_selection   import train_test_split
from sklearn.grid_search       import GridSearchCV
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import types

from commonModel import processData, loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--model" , type=str             )
parser.add_argument("--input" , type=str, default="" )
parser.add_argument("--query" , type=str, default="" ) 
parser.add_argument("--output", type=str, default="" )
parser.add_argument("--traint", type=str, default="" )

def testModel( Model, dataFrame ):
	import warnings
	warnings.filterwarnings('ignore')
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_values    = X_dataFrame.values; 
	Y_values = np.array( Model.predict( X_values ) )
	
	return pd.DataFrame(data=Y_values, index=index, columns=['price'] )

args = parser.parse_args()

modelFileName  = args.model
inputFileName  = args.input
outputFileName = args.output 

#trainFilename  = args.train

#Load a trained model
Model = None
with open( modelFileName, 'rb') as fid:
	Model = cPickle.load(fid)
#Read data
inputDataFrame = None
if args.query != "" and args.input == "":
	query = eval( "dict({})".format( args.query ) ) 
	inputDataFrame = pd.DataFrame( data=query, index=[0] )
if args.input != "" and args.query == "":
	inputDataFrame = loadData( inputFileName  )

inputDataFrame = processData( inputDataFrame )

if inputDataFrame.size > 1:
	predictedData  = testModel( Model, inputDataFrame )
	
	if outputFileName == "":
		for index, row in predictedData.iterrows():
			price = row.price
			print("{:,}".format( int( price ) ) )
	else :
		predictedData.to_csv( outputFileName, index_label='index', sep=';' )

