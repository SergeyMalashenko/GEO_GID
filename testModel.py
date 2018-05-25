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
parser.add_argument("--test"  , type=str             )
parser.add_argument("--input" , type=str             )
parser.add_argument("--output", type=str, default="" )

def testModel( testedModel, dataFrame, targetColumn ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	Y_predict = testedModel.predict( X_values )
	
	print( "Errors on the validation set" )
	print( "mean square:     ", mean_squared_error   ( Y_values, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_values, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_values, Y_predict ) )
	
	Y_predict = np.array( Y_predict )
	Y_values  = np.array( Y_values  )
	
	Y_result = np.abs( Y_predict - Y_values )/Y_values

	for threshold in [0.025, 0.050, 0.100 ]:
		bad_s  = len( Y_result[Y_result  > threshold ] )
		good_s = len( Y_result[Y_result <= threshold ] )
		print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, bad_s/(good_s+bad_s)) )
	return

args = parser.parse_args()

testFileName  = args.test
modelFileName = args.input

#Load a trained model
with open( modelFileName, 'rb') as fid:
	testedModel = cPickle.load(fid)

testDataFrame = loadData( testFileName  )

testModel( testedModel, testDataFrame, TARGET_COLUMN )

