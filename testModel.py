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

parser.add_argument("--analysis"  , action="store_true"     )

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

def getTopKClosestItems( inputItem, closestItem_s, PREPROCESSOR_X, MODEL_FEATURE_NAMES, topk=5 ) :
	if not closestItem_s.empty :
		droppedField_s = ['total_square','living_square','kitchen_square']
		droppedIndex_s = [index for index,field in enumerate(MODEL_FEATURE_NAMES) if field in droppedField_s ]
		
		processedInputItem     = inputItem    [ MODEL_FEATURE_NAMES ]
		processedClosestItem_s = closestItem_s[ MODEL_FEATURE_NAMES ]
		
		processedInputItem_numpy     = processedInputItem    .values
		processedClosestItem_s_numpy = processedClosestItem_s.values
		processedInputItem_numpy     = processedInputItem_numpy.reshape(1,-1) 
	 	
		processedInputItem_numpy     = PREPROCESSOR_X.transform( processedInputItem_numpy     ); processedInputItem_numpy     = np.delete(processedInputItem_numpy    , droppedIndex_s, axis=1 )
		processedClosestItem_s_numpy = PREPROCESSOR_X.transform( processedClosestItem_s_numpy ); processedClosestItem_s_numpy = np.delete(processedClosestItem_s_numpy, droppedIndex_s, axis=1 )
		processedResult_s_numpy      = processedClosestItem_s_numpy - processedInputItem_numpy
		processedResult_s_numpy      = np.linalg.norm( processedResult_s_numpy, axis=1 )
		
		index_s = processedResult_s_numpy.argsort()[:topk]
		return closestItem_s.iloc[index_s]
	else :
		return closestItem_s

ALPHA_BARGAINING   = 0.96
ALPHA_FLOOR_NUMBER = np.array([
	[ 1.00, 0.95, 0.97 ],
	[ 1.05, 1.00, 1.02 ],
	[ 1.03, 0.98, 1.00 ]
])
ALPHA_APARTMENT_CONDITION = np.array([
	[     0, -4376, -7146, -11092, -15507 ],
	[  4376,     0, -2771,  -6716, -11131 ],
	[  7146,  2771,     0,  -3945,  -8361 ],
	[ 11092,  6716,  3945,      0,  -4415 ],
	[ 15507, 11131,  8361,   4415,      0 ]
])

def processClosestItems( inputItem, closestItem_s, predictedPrice, verboseFlag=False ):
	RESULT_PRICE_S = dict();
	RESULT_PRICE_S['predictedPrice'] = int( predictedPrice.values[0]                                   ) 
	RESULT_PRICE_S['medianPrice'   ] = int( 0 ) 
	RESULT_PRICE_S['meanPrice'     ] = int( 0 ) 
	RESULT_PRICE_S['maxPrice'      ] = int( 0 )
	RESULT_PRICE_S['minPrice'      ] = int( 0 )
	
	RESULT_ALPHA_S = dict()
	RESULT_ALPHA_S["closestObjectsPrice"         ] = list()
	RESULT_ALPHA_S["closestObjectsSquare"        ] = list()
	RESULT_ALPHA_S["closestObjectsPricePerSquare"] = list()
	RESULT_ALPHA_S["BargainingCorrection"        ] = list()
	RESULT_ALPHA_S["FloorNumberCorrection"       ] = list()
	RESULT_ALPHA_S["ApartmentConditionCorrection"] = list()
	RESULT_ALPHA_S["ResultPrice"                 ] = int(0)
	
	if not closestItem_s.empty :
		#Calculate required prices
		#inputPrice           = predictedPrice.values[0]  
		inputSquare          = inputItem['total_square'    ]
		inputFloorNumber     = inputItem['floor_number'    ]
		inputNumberOfFloors  = inputItem['number_of_floors']
		#inputPricePerSquare  = inputPrice/inputSquare
		
		closestPrice_s          = np.array( list(map( float, closestItem_s['price'           ].values )) )
		closestSquare_s         = np.array( list(map( float, closestItem_s['total_square'    ].values )) )
		closestFloorNumber_s    = np.array( list(map( float, closestItem_s['floor_number'    ].values )) )
		closestNumberOfFloors_s = np.array( list(map( float, closestItem_s['number_of_floors'].values )) )
		closestPricePerSquare_s = closestPrice_s/closestSquare_s
		
		pricePerSquareMedian = np.median( closestPricePerSquare_s )
		pricePerSquareMean   = np.mean  ( closestPricePerSquare_s )
		pricePerSquareMax    = np.max   ( closestPricePerSquare_s )
		pricePerSquareMin    = np.min   ( closestPricePerSquare_s )
		
		RESULT_PRICE_S['predictedPrice'] = int( predictedPrice.values[0]                                   ) 
		RESULT_PRICE_S['medianPrice'   ] = int( pricePerSquareMedian*inputSquare ) 
		RESULT_PRICE_S['meanPrice'     ] = int( pricePerSquareMean  *inputSquare ) 
		RESULT_PRICE_S['maxPrice'      ] = int( pricePerSquareMax   *inputSquare )
		RESULT_PRICE_S['minPrice'      ] = int( pricePerSquareMin   *inputSquare )
		
		#Calculate required coeffecients
		inputPrice = int( pricePerSquareMedian*inputItem[['total_square']].values[0]) 
		inputPricePerSquare  = inputPrice/inputSquare
		
		inputFloorStatus     = 1
		inputFloorStatus     = 0 if inputFloorNumber == 1                   else inputFloorStatus
		inputFloorStatus     = 2 if inputFloorNumber == inputNumberOfFloors else inputFloorStatus
		
		ALPHA_BARGAINING_S   = np.full( closestPrice_s.shape, ALPHA_BARGAINING                        )
		ALPHA_FLOOR_NUMBER_S = np.full( closestPrice_s.shape, ALPHA_FLOOR_NUMBER[inputFloorStatus][1] )
		ALPHA_FLOOR_NUMBER_S[ closestFloorNumber_s == 1                       ] = ALPHA_FLOOR_NUMBER[inputFloorStatus][0] 
		ALPHA_FLOOR_NUMBER_S[ closestFloorNumber_s == closestNumberOfFloors_s ] = ALPHA_FLOOR_NUMBER[inputFloorStatus][2] 
		
		inputPricePerSquare     = inputPricePerSquare     * ALPHA_BARGAINING
		closestPricePerSquare_s = closestPricePerSquare_s * ALPHA_BARGAINING 
		closestPricePerSquare_s = closestPricePerSquare_s * ALPHA_FLOOR_NUMBER_S
		
		deltaPricePerSquare_s = inputPricePerSquare - closestPricePerSquare_s
		
		forward_index_s  = np.argsort( abs( deltaPricePerSquare_s ) )
		backward_index_s = np.full( forward_index_s.shape, 0 ) 
		for i, index in enumerate( forward_index_s ):
			backward_index_s[index] = i
		fltDeltaPricePerSquare_s = deltaPricePerSquare_s[ forward_index_s ]
		intDeltaPricePerSquare_s = np.zeros((ALPHA_APARTMENT_CONDITION.shape[0], len(closestItem_s) ))
		flt2intError_s           = np.zeros( ALPHA_APARTMENT_CONDITION.shape[0] )
		
		for i in range(ALPHA_APARTMENT_CONDITION.shape[0]):
			AlphaApartmentCondition = ALPHA_APARTMENT_CONDITION[i]
			flt2intError            = 0.
			for j in range( len(closestItem_s) ):
				fltDelta  = fltDeltaPricePerSquare_s[j]
				Index     = np.argmin(np.abs( fltDelta - AlphaApartmentCondition + flt2intError )) 
				intDelta  = AlphaApartmentCondition[ Index ]
				intDeltaPricePerSquare_s[i][j] = intDelta
				flt2intError += fltDelta - intDelta
			flt2intError_s[i] = flt2intError
		resDeltaPricePerSquare_s = intDeltaPricePerSquare_s[ np.argmin(flt2intError_s) ][ backward_index_s ]
	
		ALPHA_APP_CONDITION_S = resDeltaPricePerSquare_s/closestPricePerSquare_s 
		
		RESULT_ALPHA_S = dict()
		RESULT_ALPHA_S["closestObjectsPrice"         ] = closestPrice_s         .tolist()
		RESULT_ALPHA_S["closestObjectsSquare"        ] = closestSquare_s        .tolist()
		RESULT_ALPHA_S["closestObjectsPricePerSquare"] = closestPricePerSquare_s.tolist()
		
		RESULT_ALPHA_S["BargainingCorrection"        ] = ALPHA_BARGAINING_S     .tolist()
		RESULT_ALPHA_S["FloorNumberCorrection"       ] = ALPHA_FLOOR_NUMBER_S   .tolist()
		RESULT_ALPHA_S["ApartmentConditionCorrection"] = ALPHA_APP_CONDITION_S  .tolist()
		
		closestPricePerSquare_s += resDeltaPricePerSquare_s
		RESULT_ALPHA_S["ResultPrice"] = int(np.mean(closestPricePerSquare_s)*inputSquare/10000)*10000
	
	return RESULT_PRICE_S, RESULT_ALPHA_S

def testNeuralNetworkModel( Model, preprocessorX, preprocessorY, dataFrame, droppedColumns=[], threshold=0.1 ):
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

inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsFileName )
inputDataSize  = len( inputDataFrame.index ) 

if inputDataSize > 0: # Check that input data is correct
	for i in range( inputDataSize ) :
		inputItem         = inputDataFrame                       .iloc[i]
		inputItemForModel = inputDataFrame[ MODEL_FEATURE_NAMES ].iloc[i]
		
		predicted_Y, predicted_dYdX, predicted_dX = testNeuralNetworkModel   ( REGRESSION_MODEL, PREPROCESSOR_X, PREPROCESSOR_Y, inputItemForModel )
		predicted_dYdX.columns = MODEL_FEATURE_NAMES; predicted_dX  .columns = MODEL_FEATURE_NAMES
		
		predicted_dX.sort_values( by=0, axis=1, ascending=False, inplace=True )
		
		inputTolerances = { name : inputTolerances[name] if name in inputTolerances.keys() else abs(values[0]) for name, values in predicted_dX.iteritems() }
		del inputTolerances['number_of_rooms']; del inputTolerances['kitchen_square' ]
		
		#Get the closest items
		if verboseFlag : print( inputTolerances )
		closestItem_s = getClosestItemsInDatabase( inputItem, inputDatabase, inputTable, inputTolerances )
		closestItem_s = getTopKClosestItems( inputItem, closestItem_s, PREPROCESSOR_X, MODEL_FEATURE_NAMES, topk=topkParam )
		#Process the closest items
		RESULT_PRICE_S, RESULT_ALPHA_S = processClosestItems( inputItem, closestItem_s, predicted_Y )
		
		print( "Predicted price: {:,}".format( RESULT_PRICE_S['predictedPrice'] ) )
		print( "Median    price: {:,}".format( RESULT_PRICE_S['medianPrice'   ] ) )
		print( "Mean      price: {:,}".format( RESULT_PRICE_S['meanPrice'     ] ) )
		print( "Max       price: {:,}".format( RESULT_PRICE_S['maxPrice'      ] ) )
		print( "Min       price: {:,}".format( RESULT_PRICE_S['minPrice'      ] ) )
		
		print( "Closest objects        price: {:}".format( RESULT_ALPHA_S["closestObjectsPrice"         ] ) )
		print( "Closest objects       square: {:}".format( RESULT_ALPHA_S["closestObjectsSquare"        ] ) )
		print( "Closest objects price/square: {:}".format( RESULT_ALPHA_S["closestObjectsPricePerSquare"] ) )
		
		print( "Bargaining          corrections: {:}".format( RESULT_ALPHA_S["BargainingCorrection"        ] ) )
		print( "Floor number        corrections: {:}".format( RESULT_ALPHA_S["FloorNumberCorrection"       ] ) )
		print( "Apartment condition corrections: {:}".format( RESULT_ALPHA_S["ApartmentConditionCorrection"] ) )
		
		print( "Result price                   : {:}".format( RESULT_ALPHA_S["ResultPrice"                 ] ) )
		
		if verboseFlag :
			print( closestItem_s[['price','total_square','exploitation_start_year','created_at','floor_number','number_of_floors']] )
		else :
			print( closestItem_s[['re_id']].to_json( orient='records') )
