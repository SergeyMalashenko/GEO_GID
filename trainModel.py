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

from sklearn.pipeline            import Pipeline

import math

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import pydot

import torch
import torch.optim

from commonModel import loadCSVData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN, QuantileRegressionLoss, HuberRegressionLoss
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles
from commonModel import LinearNet

import matplotlib
#matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--input"  , type=str             )
parser.add_argument("--model"  , type=str, default="" )
parser.add_argument("--seed"   , type=int, default=43 )
parser.add_argument("--output" , type=str, default="" )
parser.add_argument("--limits" , type=str, default="" )

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
	
	dataFrame = excludeAnomalies         ( dataFrame, targetColumn )
	dataFrame = selectFeatures           ( dataFrame, targetColumn )
	
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.width', 175 ):
		print( dataFrame.describe() )
	
	return dataFrame
"""
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
"""
"""
def visualizeRandomForestTree( Model ):
	tree = Model.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')
	# Write graph to a png file
	graph.write_png('tree.png')
"""
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
	
	return clf, (Y_predict, Y_test)

def trainGradientBoostingModel( dataFrame, targetColumn, seed, tolerance=0.25 ):
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

	#clf = GradientBoostingRegressor(n_estimators = 400, max_depth = 15, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')	
	estimator = GradientBoostingRegressor()	
	param_grid = {'n_estimators':(500,), 'max_depth':(10,), 'subsample':(0.8,), 'learning_rate':(0.05,) }
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

def trainNeuralNetworkModel( dataFrame, targetColumn, seed=43, droppedColumns=[] ):
	dataFrame.drop( droppedColumns, axis=1, inplace=True )
	
	FEATURE_NAMES = list( dataFrame.columns ); FEATURE_NAMES.remove( targetColumn )
	COLUMNS       = list( dataFrame.columns );
	LABEL         = targetColumn;
	
	INDEX       = dataFrame.index.values
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values
	
	FEATURE_DEFAULTS = ((X_dataFrame.max()+X_dataFrame.min())*0.5).to_dict()
	
	preprocessorY = MinMaxScaler()
	#preprocessorY = StandardScaler()
	preprocessorY.fit( Y_values )
	preprocessorX = MinMaxScaler()
	#preprocessorX = StandardScaler()
	preprocessorX.fit( X_values )
	
	Y_values = preprocessorY.transform( Y_values )
	X_values = preprocessorX.transform( X_values )
	
	device = torch.device('cpu')
	#device = torch.device('cuda') # Uncomment this to run on GPU
	
	#Create model
	in_size = len( FEATURE_NAMES )
	#model = ConvolutionalNet( in_size ).to( device )
	model = LinearNet( in_size ).to( device )
	
	learning_rate = 0.001
	#loss_fn       = torch.nn.SmoothL1Loss()
	#loss_fn       = QuantileRegressionLoss( 0.5 ) 
	#loss_fn       = HuberRegressionLoss( 0.15 ) 
	#loss_fn       = torch.nn.MSELoss  ( size_average=False)
	loss_fn       = torch.nn.L1Loss  ( )
	#optimizer     = torch.optim.SGD   ( model.parameters(), lr=learning_rate, momentum=0.9)
	optimizer     = torch.optim.Adam  ( model.parameters(), lr=learning_rate )
	scheduler     = torch.optim.lr_scheduler.StepLR( optimizer, step_size=500, gamma=0.1)
	
	batch_size    = 256
	max_nbr_corrects_ = 0
	for t in range(5000):
		X_numpyTrain, X_numpyTest, Y_numpyTrain, Y_numpyTest = train_test_split( X_values, Y_values, test_size=0.2 )
		
		X_torchTrain = torch.from_numpy( X_numpyTrain.astype( np.float32 ) ).to( device )
		X_torchTest  = torch.from_numpy( X_numpyTest .astype( np.float32 ) ).to( device )
		
		Y_torchTrain = torch.from_numpy( Y_numpyTrain.astype( np.float32 ) ).to( device )
		Y_torchTest  = torch.from_numpy( Y_numpyTest .astype( np.float32 ) ).to( device )
		
		train_size     = X_numpyTrain.shape[0]
		test_size      = X_numpyTest .shape[0]
		
		train_index_s  = torch.randperm( train_size )
		X_torchTrain_s = X_torchTrain[ train_index_s ]
		Y_torchTrain_s = Y_torchTrain[ train_index_s ]
		
		test_index_s   = torch.randperm( test_size )
		X_torchTest_s  = X_torchTest [ test_index_s ]
		Y_torchTest_s  = Y_torchTest [ test_index_s ]
		
		X_torchTrain_s = torch.split( X_torchTrain, batch_size, dim=0 )
		Y_torchTrain_s = torch.split( Y_torchTrain, batch_size, dim=0 )
		
		X_torchTest_s  = torch.split( X_torchTest , batch_size, dim=0 )
		Y_torchTest_s  = torch.split( Y_torchTest , batch_size, dim=0 )
		
		for param_group in optimizer.param_groups:
			learning_rate = param_group['lr']
		#Train
		for i in range( len(Y_torchTrain_s)-1 ):
			x = X_torchTrain_s[i]
			y = Y_torchTrain_s[i]
			
			y_pred = model(x)
			loss   = loss_fn(y_pred, y)
			
			model.zero_grad()
			loss.backward  ()
			optimizer.step ()
		#Test
		loss_ = 0
		Y_torchPredict   = torch.zeros( Y_torchTest.shape, dtype=torch.float32 ).to( device )
		Y_torchPredict_s = torch.split( Y_torchPredict, batch_size, dim=0 )
		for i in range( len(Y_torchPredict_s)-1 ):
			x      = X_torchTest_s[i]
			y      = Y_torchTest_s[i]
			y_pred = model(x) 
			
			Y_torchPredict_s[i].copy_( y_pred )
			loss_ += loss_fn( y_pred, y )
		loss_ /= (len(Y_torchPredict_s)-1) 
		
		Y_numpyPredict = Y_torchPredict.detach().numpy()
		
		length = (len(Y_torchPredict_s)-1)*batch_size
		mean_squared_error_  = mean_squared_error    ( Y_numpyTest[:length], Y_numpyPredict[:length] )
		mean_absolute_error_ = mean_absolute_error   ( Y_numpyTest[:length], Y_numpyPredict[:length] )
		mean_median_error_   = median_absolute_error ( Y_numpyTest[:length], Y_numpyPredict[:length] )
		
		threshold = 0.1; eps = 0.001
		cur_nbr_corrects_ = np.sum( np.abs( (Y_numpyPredict[:length] - Y_numpyTest[:length])/( Y_numpyTest[:length] + eps ) < threshold ) )
		max_nbr_corrects_ = max( cur_nbr_corrects_, max_nbr_corrects_ )
		print( "epoch: {:6d}, loss: {:6.4f}, mean squared error: {:6.4f}, mean absolute error: {:6.4f}, mean median error: {:6.4f}, nbr_corrects=({}/{}/{})".format( t, loss_, mean_squared_error_, mean_absolute_error_, mean_median_error_, cur_nbr_corrects_, max_nbr_corrects_, length ) )
		scheduler.step()
	# Check model
	X_numpyTotal = X_values
	Y_numpyTotal = Y_values
	
	X_torchTotal = torch.from_numpy( X_numpyTotal.astype( np.float32 ) ).to( device )
	Y_torchTotal = torch.from_numpy( Y_numpyTotal.astype( np.float32 ) ).to( device )
	
	Y_torchPredict = model( X_torchTotal )
	Y_numpyPredict = Y_torchPredict.detach().numpy()
	Y_numpyTotal   = Y_torchTotal  .detach().numpy()
	
	Y_numpyPredict = preprocessorY.inverse_transform( Y_numpyPredict )
	Y_numpyTotal   = preprocessorY.inverse_transform( Y_numpyTotal   )
	
	Y_relErr = np.abs( Y_numpyPredict - Y_numpyTotal )*100/Y_numpyTotal
	for threshold in [ 2.5, 5.0, 10.0, 15.0 ]:
		bad_s  = np.sum( ( Y_relErr  > threshold ).astype( np.int ) )
		good_s = np.sum( ( Y_relErr <= threshold ).astype( np.int ) )
		print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, good_s/(good_s+bad_s)) )
	
	modelPacket = dict()
	modelPacket['model'           ] = model
	modelPacket['preprocessorX'   ] = preprocessorX
	modelPacket['preprocessorY'   ] = preprocessorY
	
	modelPacket['feature_names'   ] = FEATURE_NAMES
	modelPacket['feature_defaults'] = FEATURE_DEFAULTS
	
	#X_numpyTrain_Gradient = np.abs( model.gradient( X_torchTrain_mean ).detach().numpy() )
	#X_numpyTest_Gradient  = np.abs( model.gradient( X_torchTest_mean  ).detach().numpy() )
	#
	#print( "Importances of different features")
	#Importances = list( X_numpyTest_Gradient )
	#featureImportances = [(feature, round(importance, 2)) for feature, importance in zip( FEATURE_NAMES, Importances)]
	#featureImportances = sorted(featureImportances, key = lambda x: x[1], reverse = True)
	#[print('Variable: {:30} Importance: {:30}'.format(*pair)) for pair in featureImportances];
	return modelPacket, ( Y_numpyPredict, Y_numpyTotal )

def postProcessData( modelPacket, dataFrame, targetColumn, droppedColumns=[] ) :
	dataFrame.drop( droppedColumns, axis=1, inplace=True )
	
	FEATURE_NAMES = list( dataFrame.columns ); FEATURE_NAMES.remove( targetColumn )
	COLUMNS       = list( dataFrame.columns );
	LABEL         = targetColumn;
	
	Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
	X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
	Y_values    = Y_values
	
	preprocessorY = modelPacket['preprocessorY']
	preprocessorY.fit( Y_values )
	preprocessorX = modelPacket['preprocessorX']
	preprocessorX.fit( X_values )
	
	Y_values = preprocessorY.transform( Y_values )
	X_values = preprocessorX.transform( X_values )
	
	X_numpyTotal = X_values
	Y_numpyTotal = Y_values
	
	device = torch.device('cpu')
	#device = torch.device('cuda') # Uncomment this to run on GPU
	
	X_torchTotal = torch.from_numpy( X_numpyTotal.astype( np.float32 ) ).to( device )
	Y_torchTotal = torch.from_numpy( Y_numpyTotal.astype( np.float32 ) ).to( device )
	
	Y_torchPredict = modelPacket['model']( X_torchTotal )
	Y_numpyPredict = Y_torchPredict.detach().numpy()
	Y_numpyTotal   = Y_torchTotal  .detach().numpy()
	
	Y_numpyPredict = preprocessorY.inverse_transform( Y_numpyPredict )
	Y_numpyTotal   = preprocessorY.inverse_transform( Y_numpyTotal   )
	
	plt.subplot(2, 1, 1)
	
	threshold = 10
	
	Y_relativeError = np.abs( Y_numpyPredict - Y_numpyTotal )*100/Y_numpyTotal
	pricePerSquare  = ( dataFrame.price / dataFrame.total_square ).values.reshape(-1,1)
	
	allValues = pricePerSquare
	mask = Y_relativeError > threshold
	badValues  = pricePerSquare[ mask ] 
	mask = Y_relativeError <= threshold
	goodValues = pricePerSquare[ mask ]
	
	
	bins = range(30)
	bins = [i * 0.5e4 for i in bins]
	
	plt.hist([ allValues, goodValues, badValues ], bins=bins, histtype='bar', color=['green','yellow','red'])	
		
	plt.subplot(2, 1, 2)
	plt.plot   ( Y_numpyTotal  , Y_numpyTotal, c='blue' )
	plt.plot   ( Y_numpyTotal  , Y_numpyTotal*(1.0 + 0.1), c='red'  )
	plt.plot   ( Y_numpyTotal  , Y_numpyTotal*(1.0 - 0.1), c='red'  )
	plt.scatter( Y_numpyPredict, Y_numpyTotal )
	plt.show()
	
	return
 
inputFileName  = args.input
modelFileName  = args.model
outputFileName = args.output
limitsFileName = args.limits
seed           = args.seed

trainDataFrame = loadCSVData                     ( inputFileName  )
trainDataFrame = limitDataUsingLimitsFromFilename( trainDataFrame, limitsFileName )
trainDataFrame = limitDataUsingProcentiles       ( trainDataFrame )

trainDataFrame = preProcessData( trainDataFrame, TARGET_COLUMN, seed )

#TrainedModelPacket, ( Y_predict, Y_test ) = trainRandomForestModel    ( trainDataFrame, TARGET_COLUMN, seed )
#TrainedModelPacket, ( Y_predict, Y_test ) = trainGradientBoostingModel( trainDataFrame, TARGET_COLUMN, seed )

trainedModelPacket, ( Y_predict, Y_test ) = trainNeuralNetworkModel   ( trainDataFrame, TARGET_COLUMN, seed )

postProcessData( trainedModelPacket, trainDataFrame, TARGET_COLUMN )

if modelFileName != "" :
	with open( modelFileName, 'wb') as fid:
		cPickle.dump( trainedModelPacket, fid)

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
