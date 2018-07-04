import pandas            as pd
import numpy             as np

from sklearn.preprocessing      import LabelEncoder
from sklearn.preprocessing      import MinMaxScaler
from sklearn.preprocessing      import minmax_scale
from sklearn.preprocessing      import MaxAbsScaler
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import RobustScaler
from sklearn.preprocessing      import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

import json
import torch
import torch.optim

FLOAT_COLUMNS = [ 'price', 'longitude', 'latitude', 'total_square', 'living_square', 'kitchen_square']
INT_COLUMNS   = [ 'number_of_rooms', 'floor_number', 'number_of_floors', 'exploitation_start_year' ]
STR_COLUMNS   = [ 'type', 'bulding_type' ]
TARGET_COLUMN =   'price'

def check_float( x ):
	try:
		float(x)
	except ValueError:
		return False
	return True

def check_row( row ):
	#check_float_s = check_float( row.longitude ) and check_float( row.latitude ) and check_float( row.exploitation_start_year )
	check_float_s = check_float( row.longitude ) and check_float( row.latitude )
	if 'exploitation_start_year' in row : check_float_s = check_float_s and check_float( row.exploitation_start_year )

	return check_float_s

class QuantileRegressionLoss( torch.nn.Module ):
	def __init__(self, q):
		super(QuantileRegressionLoss,self).__init__()
		self.q = q
	def forward(self, input, target):
		e = ( input - target )
		result = torch.mean( torch.max( self.q*e, (self.q-1)*e ))
		return result

def limitDataUsingLimitsFromFilename( dataFrame, limitsFileName ) :
	limitsData = dict()
	with open( limitsFileName ) as f:
		limitsData = json.load(f)
	mask = True
	for columnName in limitsData.keys() :
		MIN_VALUE = limitsData[ columnName ]['min']
		MAX_VALUE = limitsData[ columnName ]['max']
		mask = (dataFrame[ columnName ] >= MIN_VALUE ) & (dataFrame[ columnName ] <= MAX_VALUE ) & mask
	
	dataFrame = dataFrame[ mask ]
	
	dataFrame.drop(labels=['kitchen_square','living_square','floor_number'], axis=1, inplace=True)
		
	return dataFrame

def limitDataUsingProcentiles( dataFrame ):
	if 'price' in dataFrame.columns :
		mask = True
		
		pricePerSquare       = ( dataFrame['price']/dataFrame['total_square'] )
		pricePerSquareValues = pricePerSquare.values
		
		robustScaler = RobustScaler(quantile_range=(15, 85) )
		robustScaler.fit( pricePerSquareValues.reshape((-1,1)) )
		pricePerSquareValues = robustScaler.transform( pricePerSquareValues.reshape((-1,1)) ).reshape(-1)
		
		mask = ( pricePerSquareValues > -1 ) & ( pricePerSquareValues  < 1 ) & mask
		
		dataFrame = dataFrame[ mask ]	
	
	return dataFrame
	
def loadCSVData( fileName, COLUMN_TYPE='NUMERICAL' ): # NUMERICAL, OBJECT, ALL

	dataFrame = pd.read_csv(
		fileName, 
		sep=";",
		encoding='cp1251', 
		#verbose=True, 
		keep_default_na=False
	).dropna(how="all")
	
	if 'price' in dataFrame.columns : dataFrame = dataFrame[ dataFrame['price'].apply( check_float ) ]
	dataFrame = dataFrame[ dataFrame.apply( check_row  , axis=1 ) ]
	
	if 'price'                   in dataFrame.columns :  dataFrame['price'                  ] = dataFrame['price'                  ].astype(np.float64)
	if 'exploitation_start_year' in dataFrame.columns :  dataFrame['exploitation_start_year'] = dataFrame['exploitation_start_year'].astype(np.float64)
	dataFrame['longitude' ] = dataFrame['longitude' ].astype(np.float64)
	dataFrame['latitude'  ] = dataFrame['latitude'  ].astype(np.float64)
	
	#print('Shape of the data with all features:', dataFrame.shape)
	if COLUMN_TYPE == 'NUMERICAL' :
		dataFrame = dataFrame.select_dtypes(exclude=['object'])
	#if COLUMN_TYPE == 'OBJECT'    :
	#	dataFrame = dataFrame.select_dtypes(exclude=['number'])
	#print('Shape of the data with numerical features:', dataFrame.shape)
	#print("List of features contained our dataset:",list( dataFrame.columns ))
	
	subset = None
	if 'price' in dataFrame.columns : 
		subset=['price', 'total_square', 'number_of_rooms' ]	
	else :
		subset=['total_square', 'number_of_rooms' ]	
	dataFrame.drop_duplicates(subset=subset, keep='first', inplace=True)	
	
	return dataFrame
