#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest

from sklearn.model_selection   import train_test_split
from sklearn.model_selection   import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble          import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing     import QuantileTransformer
from sklearn.preprocessing     import LabelEncoder
from sklearn.preprocessing     import MinMaxScaler, StandardScaler
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.tree              import export_graphviz

from sklearn.svm import SVR

#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline            import Pipeline

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.layers import BatchNormalization 

#import keras
import math

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import pydot

from commonModel import loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--input"  , type=str             )
parser.add_argument("--model"  , type=str, default="" )
parser.add_argument("--seed"   , type=int, default=0  )
parser.add_argument("--output" , type=str, default="" )

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
	
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.width', 175 ):
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

def trainRandomForestModel( dataFrame, targetColumn, seed, tolerance=0.25 ):
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
	param_grid = {'n_estimators':(256,), 'oob_score':(True,False),'max_features':(2,3,4,5), 'random_state':(seed,), 'bootstrap': (True, ), 'criterion':('mse',)  }
	n_jobs     = 3
	
	trainDataFrame   = dataFrame.loc[ INDEX_train ]
	priceTrain       = trainDataFrame.price
	totalSquareTrain = trainDataFrame.total_square
	
	sample_weight        = np.ones( len( trainDataFrame.index ) )
	pricePerSquareValues = priceTrain.values/totalSquareTrain.values
	pricePerSquareMedian = np.median( pricePerSquareValues )
	pricePerSquareMean   = np.mean  ( pricePerSquareValues )
	
	clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=3, fit_params={'sample_weight': sample_weight} )
	clf.fit( X_train, Y_train ); print( clf.best_params_ )
	Y_predict = clf.predict( X_test )
	
	print( "Importances of different features")
	Importances = list( clf.best_estimator_.feature_importances_)
	featureImportances = [(feature, round(importance, 2)) for feature, importance in zip( FEATURES, Importances)]
	featureImportances = sorted(featureImportances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in featureImportances];
	
	print( "Errors on the validation set" )
	print( "model score:     ", clf.score            ( X_test, Y_test    ) )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )
	
	return clf, None

def trainSubModel( dataFrame, targetColumn, seed ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	INDEX       = dataFrame.index.values
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	
	preprocessorY = MinMaxScaler()
	preprocessorY.fit( Y_values )
	preprocessorX = StandardScaler()
	preprocessorX.fit( X_values )
	
	Y_values = preprocessorY.transform( Y_values )
	X_values = preprocessorX.transform( X_values )
	
	X_train, X_test, Y_train, Y_test, INDEX_train, INDEX_test = train_test_split( X_values, Y_values, INDEX, test_size=0.2, random_state=seed )
	
	#estimator  = RandomForestRegressor()
	#param_grid = {'n_estimators':(16,32,64,), 'oob_score':(True,False),'max_features':(2,3), 'random_state':(seed,), 'bootstrap': (True, ) }
	#n_jobs = 3
	#clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=4 )
	
	#Create model
	model = Sequential()
	model.add( Dense             (  256, input_dim=2, kernel_initializer='uniform', activation='relu' ))
	model.add( Dense             (  512,              kernel_initializer='uniform', activation='relu' ))
	model.add( Dense             (  1024,              kernel_initializer='uniform', activation='relu' ))
	#model.add( Dense             (  1024,              kernel_initializer='uniform', activation='relu' ))
	model.add( Dense             (     1  ))
	#Compile model
	sgd  = keras.optimizers.SGD (lr=0.01, momentum=0.0, decay=0.0, nesterov=False  )
	adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile( loss='mse', optimizer=sgd, metrics=['accuracy',] )
	
	def scheduler(epoch):
		initial_lrate = 0.01
		drop          = 0.5
		epochs_drop   = 10.0
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
		print( lrate )
		return lrate
	change_lr = keras.callbacks.LearningRateScheduler(scheduler)
	# Fit the model
	model.fit(X_train, Y_train, epochs=30, batch_size=10, callbacks=[change_lr,], validation_data=( X_test, Y_test) )
	
	#clf.fit( X_train, Y_train );
	Y_predict = model.predict( X_test )
	
	print( "Errors on the validation set" )
	#print( "model score:     ", model.score            ( X_test, Y_test    ) )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )
	
	#postProcessData( INDEX_test, X_test, Y_test, Y_predict ) 
	return model, ( Y_predict, Y_test ) 

def trainFullModel( dataFrame, targetColumn, seed, tolerance=0.25 ):
	import warnings
	warnings.filterwarnings('ignore')
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	#subDataFrame                   = dataFrame[['latitude','longitude','number_of_rooms']].copy()
	subDataFrame                   = dataFrame[['latitude','longitude']]
	subDataFrame['PricePerSquare'] = dataFrame['price']/dataFrame['total_square']
	
	subModelPricePerSquare, (Y_predict,Y_test) = trainSubModel( subDataFrame, 'PricePerSquare', seed )
	
	plt.plot   ( Y_test, Y_test    )
	plt.scatter( Y_test, Y_predict )
	plt.show()
	
	return
	"""
	INDEX    = dataFrame.index.values
	dataFrame['PricePerSquare'] = PricePerSquare
	#dataFrame['EstimatedPrice'] = PricePerSquare*dataFrame['total_square']
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	X_train, X_test, Y_train, Y_test, INDEX_train, INDEX_test = train_test_split( X_values, Y_values, INDEX, test_size=0.15, random_state=seed )
	
	estimator  = RandomForestRegressor()
	param_grid = {'n_estimators':(256,), 'oob_score':(True,False),'max_features':(2,3,4,5), 'random_state':(seed,), 'bootstrap': (True, ), 'criterion':('mse',)  }
	n_jobs     = 3
	
	fullModelPrice = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=4 )
	fullModelPrice.fit( X_train, Y_train ); print( fullModelPrice.best_params_ )
	Y_predict = fullModelPrice.predict( X_test )
	
	print( "Importances of different features")
	Importances = list( fullModelPrice.best_estimator_.feature_importances_)
	featureImportances = [(feature, round(importance, 2)) for feature, importance in zip( FEATURES, Importances)]
	featureImportances = sorted(featureImportances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in featureImportances];
	
	print( "Errors on the validation set" )
	print( "model score:     ", fullModelPrice.score ( X_test, Y_test    ) )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )
	
	postProcessData( INDEX_test, X_test, Y_test, Y_predict ) 
	
	return fullModelPrice, subModelPricePerSquare, None
	"""
def trainGradientBoostingModel( dataFrame, targetColumn, seed, tolerance=0.25 ):
	import warnings
	warnings.filterwarnings('ignore')
	
	dataFrame.drop(labels=['floor_number'], axis=1, inplace=True)
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	INDEX       = dataFrame.index.values
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values.ravel()
		
	X_train, X_test, Y_train, Y_test, INDEX_train, INDEX_test = train_test_split( X_values, Y_values, INDEX, test_size=0.15, random_state=seed )

	#clf = GradientBoostingRegressor(n_estimators = 400, max_depth = 15, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')	
	estimator = GradientBoostingRegressor()	
	param_grid = {'n_estimators':(400,), 'max_depth':(13,), 'min_samples_split':(2,), 'learning_rate':(0.1,), 'loss':('ls',) }
	n_jobs     = 1
	
	clf = GridSearchCV( estimator, param_grid, n_jobs=n_jobs, cv=3 )
	clf.fit( X_train, Y_train );
	Y_predict = clf.predict( X_test )
	
	print( "Importances of different features")
	Importances = list( clf.best_estimator_.feature_importances_)
	featureImportances = [(feature, round(importance, 2)) for feature, importance in zip( FEATURES, Importances)]
	featureImportances = sorted(featureImportances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in featureImportances];
	
	print( "Errors on the validation set" )
	print( "model score:     ", clf.score ( X_test, Y_test    ) )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "median_absolute: ", median_absolute_error( Y_test, Y_predict ) )
	
	postProcessData( INDEX_test, X_test, Y_test, Y_predict ) 
	return clf, ( Y_predict, Y_test )

def trainNeuralNetworkModel( dataFrame, targetColumn, seed, tolerance=0.25 ):
	dataFrame.drop(['floor_number','number_of_floors'], axis=1, inplace=True )
	
	FEATURES = list( dataFrame.columns ); FEATURES.remove( targetColumn )
	COLUMNS  = list( dataFrame.columns );
	LABEL    = targetColumn;
	
	INDEX       = dataFrame.index.values
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values
	
	#preprocessorY = MinMaxScaler()
	preprocessorY = StandardScaler()
	preprocessorY.fit( Y_values )
	#preprocessorX = MinMaxScaler()
	preprocessorX = StandardScaler()
	preprocessorX.fit( X_values )
	
	Y_values = preprocessorY.transform( Y_values )
	X_values = preprocessorX.transform( X_values )
	
	X_train, X_test, Y_train, Y_test, INDEX_train, INDEX_test = train_test_split( X_values, Y_values, INDEX, test_size=0.1, random_state=seed )
	
	#Create model
	model = Sequential()
	model.add( Dense             (   8, input_dim=6, kernel_initializer='normal', activation='tanh' ))
	model.add( Dense             (   8,              kernel_initializer='normal', activation='tanh' ))
	model.add( Dense             (   4,              kernel_initializer='normal', activation='tanh' ))
	model.add( Dense             (   4,              kernel_initializer='normal', activation='tanh' ))
	model.add( Dense             (   1,              kernel_initializer='normal'                    ))
	
	#Compile model
	sgd  = keras.optimizers.SGD (lr=0.01, momentum=0.0, decay=0.0, nesterov=False  )
	adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile( loss='mse', optimizer=sgd, metrics=['accuracy',] )
	
	def scheduler(epoch):
		initial_lrate = 0.01
		drop          = 0.5
		epochs_drop   = 30.0
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
		print( lrate )
		return lrate
	change_lr = keras.callbacks.LearningRateScheduler(scheduler)
	# Fit the model
	model.fit(X_train, Y_train, epochs=200, batch_size=20, callbacks=[change_lr,], validation_data=( X_test, Y_test) )
	# Check the model
	Y_predict = model.predict( X_test )
	
	Y_predict = preprocessorY.inverse_transform( Y_predict )
	Y_test    = preprocessorY.inverse_transform( Y_test    )
	
	print( "Errors on the validation set" )
	print( "mean square:     ", mean_squared_error   ( Y_test, Y_predict ) )
	print( "mean absolute:   ", mean_absolute_error  ( Y_test, Y_predict ) )
	print( "mean median:     ", median_absolute_error  ( Y_test, Y_predict ) )
	
	return model, None

inputFileName  = args.input
modelFileName  = args.model
outputFileName = args.output
seed           = args.seed

trainDataFrame = loadData      ( inputFileName                 )

trainDataFrame = preProcessData( trainDataFrame, TARGET_COLUMN, seed )
#TrainedModel, ( Y_predict, Y_test ) = trainFullModel            ( trainDataFrame, TARGET_COLUMN, seed )
#TrainedModel, wrongPredictedDataFrame = trainRandomForestModel    ( trainDataFrame, TARGET_COLUMN, seed )
TrainedModel, ( Y_predict, Y_test ) = trainGradientBoostingModel( trainDataFrame, TARGET_COLUMN, seed )
#TrainedModel, wrongPredictedDataFrame = trainNeuralNetworkModel( trainDataFrame, TARGET_COLUMN, seed )

plt.plot   (    Y_test, Y_test )
plt.scatter( Y_predict, Y_test )
plt.show()

if modelFileName != "" :
	with open( modelFileName, 'wb') as fid:
		cPickle.dump( TrainedModel, fid)

#if outputFileName != "":
#	wrongPredictedDataFrame.to_csv(
#		outputFileName,
#		sep=";",
#		encoding='cp1251',
#		index=False 
#	)
#else:
#	with pd.option_context('display.max_rows', None, 'display.max_columns', 11, 'display.width', 175 ):
#		print( wrongPredictedDataFrame )
