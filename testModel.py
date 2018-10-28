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
import math

from sqlalchemy  import create_engine

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

from commonModel import LinearNet

parser = argparse.ArgumentParser()
parser.add_argument("--model"     , type=str                )
parser.add_argument("--query"     , type=str                ) 
parser.add_argument("--limits"    , type=str  , default=""  )

parser.add_argument("--tolerances", type=str  , default=""  )
parser.add_argument("--database"  , type=str  , default=""  )
parser.add_argument("--table"     , type=str  , default=""  )

parser.add_argument("--alpha"     , type=float, default=1.0 )
parser.add_argument("--top_k"     , type=int  , default=10  )

def getClosestItemsInDatabase( inputSeries, inputDataBase, inputTable, inputTolerances ) :
	engine = create_engine( inputDataBase )
	
	inputTolerancesFields = set( inputTolerances.keys() )
	inputDataFrameFields  = set( inputSeries.index      )
	
	processedTolerances = { field : inputTolerances[field] for field in inputTolerancesFields.intersection( inputDataFrameFields ) }
	
	processedLimits = dict()
	for ( field, tolerance ) in processedTolerances.items() :
		processedLimits[field] = ( inputSeries[field] - abs( tolerance ), inputSeries[field] + abs( tolerance ) )
	sql_query  = """SELECT * FROM {} WHERE """.format( inputTable )
	sql_query += """ AND """.join( "{1} <= {0} AND {0} <= {2}".format( field, min_value, max_value ) for ( field, (min_value, max_value) ) in processedLimits.items() )	
	
	resultValues = pd.read_sql_query( sql_query, engine)
	subset=['price', 'total_square', 'number_of_rooms' ]	
	resultValues.drop_duplicates(subset=subset, keep='first', inplace=True)	
	
	resultTotalSquareValues = list( map( float, resultValues[['total_square']].values ) )
	resultTotalPriceValues  = list( map( float, resultValues[['price']].values ) )
	
	pricePerSquareValues = np.array(resultTotalPriceValues)/np.array(resultTotalSquareValues) 
	pricePerSquareMedian = np.median( pricePerSquareValues )
	pricePerSquareMean   = np.mean  ( pricePerSquareValues )
	
	#return resultValues[['re_id']], pricePerSquareMedian*inputSeries[['total_square']], pricePerSquareMean*inputSeries[['total_square']]
	return resultValues[['re_id','price','total_square','longitude','latitude','number_of_rooms','exploitation_start_year']], pricePerSquareMedian*inputSeries[['total_square']], pricePerSquareMean*inputSeries[['total_square']]

def testNeuralNetworkModel( Model, preprocessorX, preprocessorY, dataFrame, droppedColumns=[] ):
	import warnings
	warnings.filterwarnings('ignore')
	
	device = torch.device('cpu')
	
	dataFrame.drop( labels=droppedColumns, inplace=True )
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_numpy = X_dataFrame.values; 
	
	X_numpy = X_numpy.reshape(1,-1)
	X_numpy = preprocessorX.transform( X_numpy )
	
	X_torch = torch.from_numpy( X_numpy.astype( np.float32 ) ).to( device )
	Y_torch = Model( X_torch )
	Y_numpy = Y_torch.detach().numpy()
	Y_numpy = preprocessorY.inverse_transform( Y_numpy )
	Y_numpy = Y_numpy.reshape(-1)
	
	return pd.Series(data=Y_numpy, index=['price'] )

def testModel( Model, dataFrame ):
	import warnings
	warnings.filterwarnings('ignore')
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_values    = X_dataFrame.values;
	Y_values = np.array( Model.predict( X_values ) )
	
	return pd.DataFrame(data=Y_values, index=index, columns=['price'] )

args = parser.parse_args()

inputQuery      = args.query
modelFileName   = args.model

limitsFileName  = args.limits

inputDatabase   = args.database
inputTable      = args.table
inputTolerances = None if args.tolerances == "" else eval( "dict({})".format( args.tolerances ) ) 

alphaParam      = args.alpha;

#Load tne model
with open( modelFileName, 'rb') as fid:
	modelPacket = cPickle.load(fid)
	
	REGRESSION_MODEL       = modelPacket['model'           ]
	PREPROCESSOR_X         = modelPacket['preprocessorX'   ]
	PREPROCESSOR_Y         = modelPacket['preprocessorY'   ]
	
	MODEL_FEATURE_NAMES    = modelPacket['feature_names'   ]
	MODEL_FEATURE_DEFAULTS = modelPacket['feature_defaults']
	
#Process query
userQuery    = eval( "dict({})".format( inputQuery ) )
defaultQuery = MODEL_FEATURE_DEFAULTS
inputQuery = defaultQuery; inputQuery.update( userQuery )

inputDataFrame = pd.DataFrame( data=inputQuery, index=[0] )

#if 'floor_flag' in MODEL_FEATURES : 
#	mask = ( inputDataFrame['floor_number'] == 1 ) | ( inputDataFrame['floor_number'] == inputDataFrame['number_of_floors'] )
#	inputDataFrame['floor_flag'] = 1; inputDataFrame[ mask ]['floor_flag'] = 0;

inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsFileName )
inputDataSize  = len( inputDataFrame.index ) 

if inputDataSize > 0: # Check that input data is correct
	for i in range( inputDataSize ) :
		inputRow         = inputDataFrame                       .iloc[i]
		inputRowForModel = inputDataFrame[ MODEL_FEATURE_NAMES ].iloc[i]
		
		predictedValue                       = testNeuralNetworkModel   ( REGRESSION_MODEL, PREPROCESSOR_X, PREPROCESSOR_Y, inputRowForModel )
		closestItems, medianValue, meanValue = getClosestItemsInDatabase( inputRow, inputDatabase, inputTable, inputTolerances )
		
		print("Predicted value {:,}".format( int( predictedValue.price*alphaParam ) ) )
		print("Median value    {:,}".format( int( medianValue         *alphaParam ) ) )
		print("Mean value      {:,}".format( int( meanValue           *alphaParam ) ) )
		print( closestItems.to_json( orient='records') )
