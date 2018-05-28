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

from commonModel import loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--model" , type=str             )
parser.add_argument("--input" , type=str             )

def checkModel( Model, dataFrame, targetColumn ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns ); LABEL = targetColumn;

	index       = dataFrame.index; 
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	Y_predict = Model.predict( X_values )
	
	print( "Errors on the test set" )
	print( "mean square:     ", mean_squared_error   ( Y_values, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_values, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_values, Y_predict ) )
	
	Y_predict = np.array( Y_predict )
	Y_values  = np.array( Y_values  )
	
	Y_relErr = np.abs( Y_predict - Y_values )*100/Y_values
	for threshold in [ 2.5, 5.0, 10.0 ]:
		bad_s  = np.sum( ( Y_relErr  > threshold ).astype( np.int ) )
		good_s = np.sum( ( Y_relErr <= threshold ).astype( np.int ) )
		print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, bad_s/(good_s+bad_s)) )
	
	x =  X_values[:,0]; y = X_values[:,1]; c = np.minimum(  Y_relErr, 25 );
	
	plt.scatter (x, y, c=c,  )
	plt.colorbar()
	plt.show    ()
	
	return pd.DataFrame(data=Y_predict, index=index, columns=['price'])

args = parser.parse_args()

modelFileName  = args.model
inputFileName  = args.input

#Load a model
Model = None
with open( modelFileName, 'rb') as fid:
	Model = cPickle.load(fid)
#Load data
inputDataFrame = loadData( inputFileName  )
#Check the model
checkModel ( Model, inputDataFrame, TARGET_COLUMN )
