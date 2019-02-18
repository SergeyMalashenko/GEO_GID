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
        predicted_dYdX .columns = MODEL_FEATURE_NAMES; predicted_dX  .columns = MODEL_FEATURE_NAMES;
        predicted_dX.sort_values( by=0, axis=1, ascending=False, inplace=True )
        
        predicted_dS = pd.DataFrame( PREPROCESSOR_X.scale_.reshape(1,-1), columns=MODEL_FEATURE_NAMES )
        
        print( "Predicted price:  {:9.2f}".format( predicted_Y .iloc[0].values[0] ) )
        predicted_dX_ = predicted_dX.iloc[0].to_dict() 
        print( "Predicted deltas:"+",".join( [ " {}={:7.4f}".format(key,value) for key,value in predicted_dX_.items() ] ) )
        predicted_dS_ = predicted_dS.iloc[0].to_dict()
        print( "Predicted scales:"+",".join( [ " {}={:7.4f}".format(key,value) for key,value in predicted_dS_.items() ] ) )
