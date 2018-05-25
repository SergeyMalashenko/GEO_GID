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

from sklearn.preprocessing     import QuantileTransformer
from sklearn.preprocessing     import LabelEncoder

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse

from commonModel import loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--train" , type=str             )
parser.add_argument("--output", type=str, default="" )
parser.add_argument("--seed"  , type=int, default=43 )

args = parser.parse_args()

def preProcessData( dataFrame, targetColumn ):
	def excludeAnomalies( dataFrame, targetColumn ):
		Y_data = dataFrame    [[ targetColumn ]];       Y_values = Y_data.values;
		X_data = dataFrame.drop( targetColumn, axis=1); X_values = X_data.values;
		clf = IsolationForest(max_samples = 200, random_state = 42); clf.fit( X_values )
		
		y_noano = clf.predict( X_values )
		y_noano = pd.DataFrame(y_noano, columns = ['Top'])
		y_noano[y_noano['Top'] == 1].index.values
		
		dataFrame = dataFrame.iloc[y_noano[y_noano['Top'] == 1].index.values]
		dataFrame.reset_index(drop = True, inplace = True)
		print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
		print("Number of rows without outliers:", dataFrame.shape[0])
		return dataFrame
	
	#def excludeAnomalies( dataFrame, targetColumn ):
	#return dataFrame
		
		
	def selectFeatures( dataFrame, targetColumn ):
		Y_data = dataFrame    [[ targetColumn ]]
		X_data = dataFrame.drop( targetColumn, axis=1)
		
		X_values = X_data.values
		Y_values = Y_data.values
		
		selection = SelectKBest  ( f_regression, k='all' )
		selector  = selection.fit( X_values, Y_values )
		
		oldFeatureNames = list( X_data.columns.values ); newFeatureNames = []
		scores = selection.scores_
		mask   = selection.get_support()
		for bool, score, featureName in zip(mask, scores, oldFeatureNames ):
			if bool:
				print( "{:17} {}".format( featureName, score ) )
				newFeatureNames.append( featureName )
		X_values_ = selector.transform( X_values )
		Y_values_ = Y_values
		
		dataFrame                 = pd.DataFrame( X_values_, columns=newFeatureNames, dtype=np.float64 )
		dataFrame[ targetColumn ] = Y_values_
		return dataFrame
	
	#for column in dataFrame:
	#	min_    = dataFrame[[column]].min   ().values[0]
	#	max_    = dataFrame[[column]].max   ().values[0]
	#	median_ = dataFrame[[column]].median().values[0]
	#	print( "{:30} min={:10}, max={:10}, median={:10}".format( column, min_, max_, median_ ) )
	
	
	dataFrame = excludeAnomalies( dataFrame, targetColumn )
	dataFrame = selectFeatures  ( dataFrame, targetColumn )
	
	for column in dataFrame:
		min_    = dataFrame[[column]].min   ().values[0]
		max_    = dataFrame[[column]].max   ().values[0]
		median_ = dataFrame[[column]].median().values[0]
		print( "{:30} min={:10}, max={:10}, median={:10}".format( column, min_, max_, median_ ) )
	
	
	return dataFrame

def trainModel( dataFrame, targetColumn ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	X_train, X_test, Y_train, Y_test = train_test_split( X_values, Y_values, test_size=0.2 )

	estimator  = RandomForestRegressor()
	param_grid = {'n_estimators':(24,28,32,36,40), 'oob_score':(True,False),'max_features':(2,3,4,5) }
	n_jobs     = 2
		
	clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs )
	clf.fit( X_train, Y_train ); print( clf.best_params_ )
	Y_predict = clf.predict( X_test )
	
	print( "Errors on the validation set" )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )

	Y_predict = np.array( Y_predict )
	Y_test    = np.array( Y_test    ).squeeze()
	
	Y_result = np.abs( Y_predict - Y_test )/Y_test
	
	for threshold in [0.025, 0.050, 0.100 ]:
		bad_s  = len( Y_result[Y_result  > threshold ] )
		good_s = len( Y_result[Y_result <= threshold ] )
		print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, bad_s/(good_s+bad_s)) )
	
	return clf

trainFileName = args.train
modelFileName = args.output

trainDataFrame = loadData      ( trainFileName                 )
trainDataFrame = preProcessData( trainDataFrame, TARGET_COLUMN )
trainedModel   = trainModel    ( trainDataFrame, TARGET_COLUMN )

if modelFileName != "" :
	with open( modelFileName, 'wb') as fid:
		cPickle.dump( trainedModel, fid)

