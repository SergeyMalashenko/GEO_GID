#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest

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

from commonModel import processData, loadData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN

parser = argparse.ArgumentParser()
parser.add_argument("--model"     , type=str               )
parser.add_argument("--input"     , type=str  , default="" )
parser.add_argument("--query"     , type=str  , default="" ) 
parser.add_argument("--output"    , type=str  , default="" )

parser.add_argument("--dataset"   , type=str  , default="" )
parser.add_argument("--tolerances", type=str  , default=""  )

def outputClosestItems( inputDataFrame, outputDataFrame, columnTolerances, fmt='plain' ):
	columns     = list( columnTolerances.keys() )
	tolerances  = list( columnTolerances.values() )
	
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
				
			if fmt == 'plain' :
				with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.width', 175 ):
					#print('->')
					#print( inputDataFrame.iloc[ i     ] )
					print('<-')
					print( resultDataFrame )
			elif fmt == 'json':
				resultDataFrame['index'] = resultDataFrame.index
				print( resultDataFrame.to_json( orient='records') )
		else:
			print("{:,}".format( 0 ) )
	
def testModel( Model, dataFrame ):
	import warnings
	warnings.filterwarnings('ignore')
	
	dataFrame.drop(labels=['floor_number'], axis=1, inplace=True)
	
	X_dataFrame = dataFrame; index = X_dataFrame.index; X_values    = X_dataFrame.values; 
	Y_values = np.array( Model.predict( X_values ) )
	
	return pd.DataFrame(data=Y_values, index=index, columns=['price'] )

args = parser.parse_args()

modelFileName  = args.model
inputFileName  = args.input
outputFileName = args.output 

#Load a trained model
Model = None
with open( modelFileName, 'rb') as fid:
	Model = cPickle.load(fid)
#Read data
inputDataFrame = None
if args.query != "" and args.input == "":
	query = eval( "dict({})".format( args.query ) ) 
	inputDataFrame = pd.DataFrame( data=query, index=[0] )
if args.input != "" and args.query == "":
	inputDataFrame = loadData( inputFileName  )
inputDataFrame = inputDataFrame[['longitude','latitude','total_square','living_square','kitchen_square','number_of_rooms','floor_number','number_of_floors']]

inputTolerances = None
if args.tolerances != "":
	inputTolerances = eval( "dict({})".format( args.tolerances ) ) 

inputDataFrame = processData( inputDataFrame )

if inputDataFrame.size > 1:
	predictedData  = testModel( Model, inputDataFrame )
	
	if outputFileName == "":
		for index, row in predictedData.iterrows():
			price = row.price
			print("{:,}".format( int( price ) ) )
	else :
		predictedData.to_csv( outputFileName, index_label='index', sep=';' )
	
	dataFileName  = args.dataset
	
	if dataFileName != "" :
		dataFrame = loadData( dataFileName )
		if dataFrame.size > 1:
			outputClosestItems( inputDataFrame, dataFrame, inputTolerances, fmt='json' )

