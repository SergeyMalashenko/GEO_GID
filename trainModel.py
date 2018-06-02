#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest

from sklearn.model_selection   import train_test_split
from sklearn.grid_search       import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

from sklearn.preprocessing     import QuantileTransformer
from sklearn.preprocessing     import LabelEncoder
from sklearn.tree              import export_graphviz

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import pydot

from commonModel import loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--input" , type=str             )
parser.add_argument("--model" , type=str, default="" )
parser.add_argument("--seed"  , type=int, default=0  )

args = parser.parse_args()

def preProcessData( dataFrame, targetColumn, seed ):
	def excludeAnomalies( dataFrame, targetColumn ):
		Y_data = dataFrame    [[ targetColumn ]];       Y_values = Y_data.values;
		X_data = dataFrame.drop( targetColumn, axis=1); X_values = X_data.values;
		clf = IsolationForest(max_samples = 200, random_state = 42); clf.fit( X_values )
		
		y_noano = clf.predict( X_values )
		y_noano = pd.DataFrame(y_noano, columns = ['Top'])
		y_noano[y_noano['Top'] == 1].index.values
		
		dataFrame = dataFrame.iloc[y_noano[y_noano['Top'] == 1].index.values]
		#dataFrame.reset_index(drop = True, inplace = True)
		print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
		print("Number of rows without outliers:", dataFrame.shape[0])
		return dataFrame
	
	def selectFeatures( dataFrame, targetColumn ):
		index  = dataFrame.index
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
		
		dataFrame                 = pd.DataFrame( X_values_, index=index, columns=newFeatureNames, dtype=np.float64 )
		dataFrame[ targetColumn ] = Y_values_
		return dataFrame
	
	dataFrame = excludeAnomalies( dataFrame, targetColumn )
	dataFrame = selectFeatures  ( dataFrame, targetColumn )
	
	print( dataFrame.describe() )
	
	return dataFrame

def postProcessData( INDEX_test, X_test, Y_test, Y_predict ) :
	threshold_s = [2.5, 5.0, 10.0, 15.0 ]
	
	Y_predict =  np.array( Y_predict )
	Y_test    =  np.array( Y_test    )
	Y_rel_err = (np.abs( Y_predict - Y_test )/Y_test*100 ).astype( np.int )
	
	for threshold in threshold_s : 
		bad_s  = np.sum( ( Y_rel_err  > threshold ).astype( np.int ) )
		good_s = np.sum( ( Y_rel_err <= threshold ).astype( np.int ) )
		print("threshold = {:5}%, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, bad_s/(good_s+bad_s)) )
	
	mask       = Y_rel_err > 25
	INDEX_test = INDEX_test[ mask ]
	X_test     = X_test    [ mask ]
	Y_test     = Y_test    [ mask ]
	Y_predict  = Y_predict [ mask ]
	
	index_s    = np.argsort( Y_test )
	INDEX_test = INDEX_test[ index_s ]
	X_test     = X_test    [ index_s ]
	Y_test     = Y_test    [ index_s ]
	Y_predict  = Y_predict [ index_s ]
	
	for i in range( INDEX_test.size ) :
		index     = INDEX_test[ i ]+2
		x_test    = X_test    [ i ]
		y_test    = Y_test    [ i ]
		y_predict = Y_predict [ i ] 
		print('{:6} {:10.1f} {:10.1f} {:10.1f}%'.format( index, y_test, y_predict, (y_predict-y_test)*100./y_test ))

def visualizeRandomForestTree( Model ):
	tree = Model.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')
	# Write graph to a png file
	graph.write_png('tree.png')

def trainModel( dataFrame, targetColumn, seed ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	INDEX       = dataFrame.index.values
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	X_train, X_test, Y_train, Y_test, INDEX_train, INDEX_test = train_test_split( X_values, Y_values, INDEX, test_size=0.15, random_state=seed )
	
	estimator  = RandomForestRegressor()
	param_grid = {'n_estimators':(64,), 'oob_score':(True,False),'max_features':(2,3,4,5), 'random_state':(seed,), 'bootstrap': (True, ), 'criterion':('mse',)  }
	n_jobs     = 3
	
	trainDataFrame   = dataFrame.loc[ INDEX_train ]
	priceTrain       = trainDataFrame.price
	totalSquareTrain = trainDataFrame.total_square
	
	sample_weight        = np.ones( len( trainDataFrame.index ) )
	pricePerSquareValues = priceTrain.values/totalSquareTrain.values
	pricePerSquareMedian = np.median( pricePerSquareValues )
	pricePerSquareMean   = np.mean  ( pricePerSquareValues )

	#sample_weight        = sample_weight
	#sample_weight        = sample_weight + np.abs(pricePerSquareValues-pricePerSquareMean)/pricePerSquareMean
	#sample_weight        = sample_weight + np.square( (pricePerSquareValues-pricePerSquareMean)/pricePerSquareMean )
	#hist_values, bin_edges = np.histogram( pricePerSquareValues )
	#hist_values = hist_values/np.sum( hist_values )
	#for i in range( pricePerSquareValues.size ):
	#	Value = pricePerSquareValues[i]
	#	for j in range( bin_edges.size - 1):
	#		if bin_edges[j+0] < Value and Value <= bin_edges[j+1] :
	#			sample_weight[i] = hist_values[j]
	
	clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=3, fit_params={'sample_weight': sample_weight} )
	#clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=3 )
	clf.fit( X_train, Y_train ); print( clf.best_params_ )
	Y_predict = clf.predict( X_test )
	
	print( "Errors on the validation set" )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )
	
	print( "Importances of different features")
	Importances = list( clf.best_estimator_.feature_importances_)
	featureImportances = [(feature, round(importance, 2)) for feature, importance in zip( FEATURES, Importances)]
	featureImportances = sorted(featureImportances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in featureImportances];
	
	postProcessData( INDEX_test, X_test, Y_test, Y_predict )
		
	return clf

def trainNeuralNetworkModel( dataFrame, targetColumn, seed ):
	import tensorflow as tf
	
	import warnings
	warnings.filterwarnings('ignore')
	
	def input_fn(data_set, pred = False):
		if pred == False:
			feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
			labels = tf.constant(data_set[LABEL].values)
			return feature_cols, labels
		if pred == True:
			feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
			return feature_cols

	tf.logging.set_verbosity(tf.logging.ERROR)
	regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
	
	regressor.fit          (input_fn=lambda: input_fn(training_set), steps=2000 )	
	ev = regressor.evaluate(input_fn=lambda: input_fn( testing_set), steps=1    )	
	
	loss_score = ev["loss"]
	print("Final Loss on the testing set: {0:f}".format(loss_score))
	
	return regressor

inputFileName = args.input
modelFileName = args.model
seed          = args.seed

trainDataFrame = loadData      ( inputFileName                 )

trainDataFrame = preProcessData( trainDataFrame, TARGET_COLUMN, seed )
Model          = trainModel    ( trainDataFrame, TARGET_COLUMN, seed )

if modelFileName != "" :
	with open( modelFileName, 'wb') as fid:
		cPickle.dump( Model, fid)

