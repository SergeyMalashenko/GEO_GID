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
import datetime
import argparse
import types
import torch
import timeit
import math
import sys

from sqlalchemy  import create_engine

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

from commonModel import LinearNet

pd.set_option('display.width'      , 500 )
pd.set_option('display.max_rows'   , 500 )
pd.set_option('display.max_columns', 500 )

parser = argparse.ArgumentParser()
parser.add_argument("--model"     , type=str                )
parser.add_argument("--query"     , type=str                ) 
parser.add_argument("--limits"    , type=str  , default=""  )

parser.add_argument("--tolerances", type=str  , default=""  )
parser.add_argument("--database"  , type=str  , default=""  )
parser.add_argument("--table"     , type=str  , default=""  )

parser.add_argument("--alpha"     , type=float, default=1.0 )
parser.add_argument("--topk"      , type=int  , default=5   )
parser.add_argument("--verbose"   , action="store_true"     )

def getClosestItemsInDatabase( inputSeries, inputDataBase, inputTable, inputTolerances, limitWithDate=False ) :
	engine = create_engine( inputDataBase )
	
	inputTolerancesFields = set( inputTolerances.keys() )
	inputDataFrameFields  = set( inputSeries.index      )
	
	processedTolerances = { field : inputTolerances[field] for field in inputTolerancesFields.intersection( inputDataFrameFields ) }
	
	processedLimits = dict()
	for ( field, tolerance ) in processedTolerances.items() :
		if tolerance != np.inf :
			processedLimits[field] = ( inputSeries[field] - abs( tolerance ), inputSeries[field] + abs( tolerance ) )
	
	sql_query  = """SELECT * FROM {} WHERE """.format( inputTable )
	sql_query += """ AND """.join( "{1} <= {0} AND {0} <= {2}".format( field, min_value, max_value ) for ( field, (min_value, max_value) ) in processedLimits.items() )	
	
	if limitWithDate == True:
		currentDate  = datetime.date.today()
		deltaDate    = datetime.timedelta(days=90)
		updatedDate  = currentDate - deltaDate
		sql_query += """ AND """ + """created_at BETWEEN '{}' AND '{}'  """.format( updatedDate, currentDate )
	
	resultValues = pd.read_sql_query( sql_query, engine)
	subset=['price', 'total_square', 'number_of_rooms' ]	
	resultValues.drop_duplicates(subset=subset, keep='first', inplace=True)	
	
	return resultValues

def processClosestItems( inputItem, closestItem_s, PREPROCESSOR_X, MODEL_FEATURE_NAMES, topk=5 ) :
	droppedField_s = ['total_square','living_square','kitchen_square']
	droppedIndex_s = [index for index,field in enumerate(MODEL_FEATURE_NAMES) if field in droppedField_s ]
	
	processedInputItem     = inputItem    [ MODEL_FEATURE_NAMES ]
	processedClosestItem_s = closestItem_s[ MODEL_FEATURE_NAMES ]
	
	processedInputItem_numpy     = processedInputItem    .values
	processedClosestItem_s_numpy = processedClosestItem_s.values
	
	processedInputItem_numpy     = processedInputItem_numpy   .reshape(1,-1) 
	#processedClosestItem_s_numpy = processedClosestItem_s_numpy.reshape(1,-1)
	
	print( processedClosestItem_s )
	 
	processedInputItem_numpy     = PREPROCESSOR_X.transform( processedInputItem_numpy     ); processedInputItem_numpy     = np.delete(processedInputItem_numpy    , droppedIndex_s, axis=1 )
	processedClosestItem_s_numpy = PREPROCESSOR_X.transform( processedClosestItem_s_numpy ); processedClosestItem_s_numpy = np.delete(processedClosestItem_s_numpy, droppedIndex_s, axis=1 )
	processedResult_s_numpy      = processedClosestItem_s_numpy - processedInputItem_numpy
	processedResult_s_numpy      = np.linalg.norm( processedResult_s_numpy, axis=1 )
	
	index_s = processedResult_s_numpy.argsort()[:topk] 
	return closestItem_s.iloc[index_s]

def testNeuralNetworkModel( Model, preprocessorX, preprocessorY, dataFrame, droppedColumns=[], threshold=0.05 ):
	import warnings
	warnings.filterwarnings('ignore')
	
	device = torch.device('cpu')
	
	dataFrame.drop( labels=droppedColumns, inplace=True )
	
	Y_scale = preprocessorY.scale_;
	X_scale = preprocessorX.scale_;
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_numpy = X_dataFrame.values; 
	
	X_numpy = X_numpy.reshape(1,-1)
	X_numpy = preprocessorX.transform( X_numpy )
	
	X_torch   = torch.from_numpy( X_numpy.astype( np.float32 ) ).to( device )
	Y_torch   = Model         ( X_torch )
	dY_torch  = Model.jacobian( X_torch )
	
	Y_base_numpy    = Y_torch  .detach().numpy()
	dYdX_base_numpy = dY_torch .detach().numpy() # \frac{df}{dx}
	
	dX_base_numpy   = (Y_base_numpy*threshold)/dYdX_base_numpy
	
	Y_numpy         = preprocessorY.inverse_transform( Y_base_numpy )
	
	dYdX_numpy      = dYdX_base_numpy
	dX_numpy        = dX_base_numpy/X_scale 
	
	#Y_numpy         = Y_numpy   .reshape(-1)
	#dYdX_numpy      = dYdX_numpy.reshape(-1)
	#dX_numpy        = dX_numpy  .reshape(-1)
	
	return pd.DataFrame( Y_numpy ), pd.DataFrame( dYdX_numpy ), pd.DataFrame( dX_numpy )

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
inputTolerances = dict() if args.tolerances == "" else eval( "dict({})".format( args.tolerances ) ) 

alphaParam      = args.alpha
topkParam       = args.topk
verboseFlag     = args.verbose

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
		
		predicted_Y, predicted_dYdX, predicted_dX = testNeuralNetworkModel   ( REGRESSION_MODEL, PREPROCESSOR_X, PREPROCESSOR_Y, inputRowForModel )
		predicted_dYdX.columns = MODEL_FEATURE_NAMES; predicted_dX  .columns = MODEL_FEATURE_NAMES
		
		predicted_dX.sort_values( by=0, axis=1, ascending=False, inplace=True )
		
		inputTolerances = { name : inputTolerances[name] if name in inputTolerances.keys() else abs(values[0]) for name, values in predicted_dX.iteritems() }
		
		#Get the closest items
		if verboseFlag : print( inputTolerances )
		closestItems = getClosestItemsInDatabase( inputRow, inputDatabase, inputTable, inputTolerances )
		#Process the closest items
		pricePerSquareMedian, pricePerSquareMean, pricePerSquareMax, pricePerSquareMin = 0, 0, 0, 0
		pricePerSquareValues = None
		if not closestItems.empty :
			closestItems = processClosestItems( inputRow, closestItems, PREPROCESSOR_X, MODEL_FEATURE_NAMES )
			
			resultTotalSquareValues = list( map( float, closestItems[['total_square']].values ) )
			resultTotalPriceValues  = list( map( float, closestItems[['price'       ]].values ) )
			
			pricePerSquareValues = np.array(resultTotalPriceValues)/np.array(resultTotalSquareValues) 
			pricePerSquareMedian = np.median( pricePerSquareValues )
			pricePerSquareMean   = np.mean  ( pricePerSquareValues )
			pricePerSquareMax    = np.max   ( pricePerSquareValues )
			pricePerSquareMin    = np.min   ( pricePerSquareValues )
		predictedPrice = int( predicted_Y.values[0]                                     ) 
		medianPrice    = int( pricePerSquareMedian*inputRow[['total_square']].values[0] ) 
		meanPrice      = int( pricePerSquareMean  *inputRow[['total_square']].values[0] ) 
		maxPrice       = int( pricePerSquareMax   *inputRow[['total_square']].values[0] )
		minPrice       = int( pricePerSquareMin   *inputRow[['total_square']].values[0] )
		
		resultPrice          = predictedPrice if minPrice < predictedPrice and predictedPrice < maxPrice else medianPrice
		pricePerSquareResult = resultPrice / inputRow[['total_square']].values[0] 
		resultWeights        = [1,1,1,1,1]
		if not closestItems.empty :
			resultWeights = ( pricePerSquareValues / pricePerSquareResult )
			#resultWeights/= resultWeights.size
			resultWeights = resultWeights.tolist()
		
		print( "Result value    {:,}".format( resultPrice    ) )
		print( "Result weights  "+" ".join("{:,.2f}".format(weight) for weight in resultWeights ) )
		
		print( "Predicted value {:,}".format( predictedPrice ) )
		print( "Median value    {:,}".format( medianPrice    ) )
		print( "Mean value      {:,}".format( meanPrice      ) )
		print( "Max value       {:,}".format( maxPrice       ) )
		print( "Min value       {:,}".format( minPrice       ) )
		
		if verboseFlag :
			print( closestItems[['price','total_square','exploitation_start_year','created_at']] )
			#print( closestItems[['price','total_square','exploitation_start_year','created_at']].to_json( orient='records') )
		else :
			print( closestItems[['re_id']].to_json( orient='records') )
		
#predicted_dYdX = predicted_dYdX.assign(features=pd.Series(MODEL_FEATURE_NAMES).values)
#predicted_dYdX = predicted_dYdX.assign(absolute=np.abs(predicted_dYdX['dYdX'].values))
#predicted_dYdX.sort_values('absolute', ascending=False, inplace=True)
#predicted_dX   = predicted_dX.assign(features=pd.Series(MODEL_FEATURE_NAMES).values)

#resultTotalSquareValues = list( map( float, resultValues[['total_square']].values ) )
#resultTotalPriceValues  = list( map( float, resultValues[['price']].values ) )
#pricePerSquareValues = np.array(resultTotalPriceValues)/np.array(resultTotalSquareValues) 
#pricePerSquareMedian = np.median( pricePerSquareValues )
#pricePerSquareMean   = np.mean  ( pricePerSquareValues )
#return resultValues, pricePerSquareMedian*inputSeries[['total_square']], pricePerSquareMean*inputSeries[['total_square']]
