#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection   import train_test_split
from sklearn.model_selection   import GridSearchCV
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import types
import torch
import timeit

from commonModel import loadCSVData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

from commonModel import LinearNet
from commonModel import ballTreeDistance

parser = argparse.ArgumentParser()
parser.add_argument("--model"     , type=str               )
parser.add_argument("--input"     , type=str  , default="" )
parser.add_argument("--query"     , type=str  , default="" ) 
parser.add_argument("--output"    , type=str  , default="" )
parser.add_argument("--limits"    , type=str  , default="" )

parser.add_argument("--tree"      , type=str  , default="" )

parser.add_argument("--dataset"   , type=str  , default="" )
parser.add_argument("--tolerances", type=str  , default="" )


def findClosestItemsUsingSearchTreeMethod( searchTreeModel, searchTreeFeatures, inputDataFrame, inputTolerances, outputDataFrame ) :
	tolerance = 0.000; 
	for feature in searchTreeFeatures : 
		tolerance = max( tolerance, inputTolerances[ feature ] )
	
	numpy_index_s = np.concatenate( searchTreeModel.query_radius( inputDataFrame[ searchTreeFeatures ].values, tolerance ) )
	numpy_index_s = np.unique( numpy_index_s )
	list_index_s  = numpy_index_s.tolist()
	return outputDataFrame.iloc[ list_index_s ]
def findClosestItemsUsingPlainMethod( inputDataFrame, inputTolerances, outputDataFrame, fmt='json' ) :
	columns     = list( inputTolerances.keys  () )
	tolerances  = list( inputTolerances.values() )
	
	totalSquare    = inputDataFrame['total_square'].values[0]
	pricePerSquare = 0 
	resultPrice    = 0
	
	for i, inputRow in inputDataFrame[columns].iterrows():
		inputValues = inputRow.values
		mask = outputDataFrame.apply( lambda row : np.all( np.abs( row[ columns ].values - inputValues ) < tolerances ), axis=1 )
		
		resultDataFrame = outputDataFrame[ mask ]
		
		if len( resultDataFrame ) > 0 :	
			pricePerSquare  = np.median( resultDataFrame['price']/resultDataFrame['total_square'] )
			resultPrice     = pricePerSquare*totalSquare 
			
			print("{:,}".format( int( resultPrice ) ) )
			
			resultDataFrame['index'] = resultDataFrame.index
			print( resultDataFrame.to_json( orient='records') )
		else:
			print("{:,}".format( 0 ) )
	
	return

def testNeuralNetworkModel( Model, preprocessorX, preprocessorY, dataFrame, droppedColumns=[] ):
	import warnings
	warnings.filterwarnings('ignore')
	
	device = torch.device('cpu')
	
	dataFrame.drop( droppedColumns, axis=1, inplace=True )
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_numpy = X_dataFrame.values; 
	
	X_numpy = preprocessorX.transform( X_numpy )
	
	X_torch = torch.from_numpy( X_numpy.astype( np.float32 ) ).to( device )
	Y_torch = Model( X_torch )
	Y_numpy = Y_torch.detach().numpy()
	Y_numpy = preprocessorY.inverse_transform( Y_numpy )
	
	return pd.DataFrame(data=Y_numpy, index=index, columns=['price'] )

def testModel( Model, dataFrame ):
	import warnings
	warnings.filterwarnings('ignore')
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_values    = X_dataFrame.values;
	Y_values = np.array( Model.predict( X_values ) )
	
	return pd.DataFrame(data=Y_values, index=index, columns=['price'] )

args = parser.parse_args()

modelFileName   = args.model
inputFileName   = args.input
outputFileName  = args.output 
limitsFileName  = args.limits
treeFileName    = args.tree
dataFileName    = args.dataset
inputTolerances = None if args.tolerances == "" else eval( "dict({})".format( args.tolerances ) ) 

#Load tne model
with open( modelFileName, 'rb') as fid:
	modelPacket = cPickle.load(fid)
	REGRESSION_MODEL = modelPacket['model'        ]
	MODEL_FEATURES   = modelPacket['features'     ]
	PREPROCESSOR_X   = modelPacket['preprocessorX']
	PREPROCESSOR_Y   = modelPacket['preprocessorY']
#Load search tree
with open( treeFileName, 'rb') as fid:
	searchTreePacket = cPickle.load(fid)
	SEARCH_TREE_MODEL    = searchTreePacket['tree'    ] 
	SEARCH_TREE_FEATURES = searchTreePacket['features']
	SEARCH_TREE_DATA     = searchTreePacket['data'    ]

#Read data
inputDataFrame = None
if args.query != "" and args.input == "":
	query = eval( "dict({})".format( args.query ) ) 
	inputDataFrame = pd.DataFrame( data=query, index=[0] )
if args.input != "" and args.query == "":
	inputDataFrame = loadCSVData( inputFileName  )
if 'floor_flag' in MODEL_FEATURES : 
	mask = ( inputDataFrame['floor_number'] == 1 ) | ( inputDataFrame['floor_number'] == inputDataFrame['number_of_floors'] )
	inputDataFrame['floor_flag'] = 1; inputDataFrame[ mask ]['floor_flag'] = 0;

inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsFileName )
inputDataFrameForModel      = inputDataFrame[ MODEL_FEATURES       ]
inputDataFrameForSearchTree = inputDataFrame[ SEARCH_TREE_FEATURES ]

if inputDataFrame.size > 1:
	predictedData  = testNeuralNetworkModel( REGRESSION_MODEL, PREPROCESSOR_X, PREPROCESSOR_Y, inputDataFrameForModel )
	
	if outputFileName == "":
		for index, row in predictedData.iterrows():
			price = row.price
			print("{:,}".format( int( price ) ) )
	else :
		predictedData.to_csv( outputFileName, index_label='index', sep=';' )
	
	dataFrame = SEARCH_TREE_DATA 
	dataFrame = findClosestItemsUsingSearchTreeMethod( SEARCH_TREE_MODEL, SEARCH_TREE_FEATURES, inputDataFrame, inputTolerances, dataFrame )
	dataFrame = findClosestItemsUsingPlainMethod     ( inputDataFrame, inputTolerances, dataFrame )
	"""
	if len( dataFrame ) > 0 :	
			pricePerSquare  = np.median( resultDataFrame['price']/resultDataFrame['total_square'] )
			resultPrice     = pricePerSquare*totalSquare 
			
			print("{:,}".format( int( resultPrice ) ) )
			
			resultDataFrame['index'] = resultDataFrame.index
			print( resultDataFrame.to_json( orient='records') )
		else:
			print("{:,}".format( 0 ) )
	"""
