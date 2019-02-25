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

#from hessian import hessian
#from hessian import jacobian

import json
import torch
import torch.optim

from sqlalchemy import create_engine

FLOAT_COLUMNS = [ 'price', 'longitude', 'latitude', 'total_square', 'living_square', 'kitchen_square', 'distance_to_metro']
#FLOAT_COLUMNS = [ 'price', 'longitude', 'latitude', 'total_square', 'living_square', 'kitchen_square']
INT_COLUMNS   = [ 'number_of_rooms', 'floor_number', 'number_of_floors', 'exploitation_start_year' ]
DATE_COLUMNS  = [ 're_created_at' ]
STR_COLUMNS   = [ 'type', 'bulding_type' ]
TARGET_COLUMN =   'price'

def check_float( x ):
	try:
		float(x)
	except ValueError:
		return False
	return True

def check_row( row ):
	check_float_s = check_float( row.longitude ) and check_float( row.latitude )
	for columnName in (FLOAT_COLUMNS + INT_COLUMNS):
		if columnName in row : check_float_s = check_float_s and check_float( row[ columnName ] )

	return check_float_s

#Huber regression
class HuberRegressionLoss( torch.nn.Module ):
	def __init__(self, delta):
		super(HuberRegressionLoss,self).__init__()
		self.eps   = 0.00001
		self.delta = delta
	def forward(self, predict, target):
		delta = self.delta
		eps   = self.eps
		
		e = ( target - predict )
		
		result = torch.zeros( e.size() )
		
		mask = torch.abs( e ).le( delta )
		result[ mask ] = 0.5*(e[ mask ]*e[ mask ]) 
		
		mask = torch.abs( e ).gt( delta )
		result[ mask ] = delta*torch.abs( e[ mask ] ) - 0.5*(delta**2)
		
		return torch.mean( result )

#Quantile regression
class QuantileRegressionLoss( torch.nn.Module ):
	def __init__(self, q):
		super(QuantileRegressionLoss,self).__init__()
		self.q = q
	def forward(self, predict, target):
		e = ( target- predict )
		result = torch.mean( torch.max( self.q*e, (self.q-1)*e ))
		return result

def limitDataUsingLimitsFromFilename( dataFrame, limitsFileName ) :
	def date_hook( json_dict ):
		for (key, value) in json_dict.items(): 
			try: 
				json_dict[key]['min'] = datetime.datetime.strptime(value['min'], "%Y-%m-%dT%H:%M:%S") 
				json_dict[key]['max'] = datetime.datetime.strptime(value['max'], "%Y-%m-%dT%H:%M:%S") 
			except: 
				pass 
		return json_dict 
	
	limitsData = dict()
	with open( limitsFileName ) as f:
		limitsData = json.load(f, object_hook=date_hook )
	mask = True
	for columnName in limitsData.keys() :
		MIN_VALUE = limitsData[ columnName ]['min']
		MAX_VALUE = limitsData[ columnName ]['max']
		
		mask = (dataFrame[ columnName ] >= MIN_VALUE ) & mask
		mask = (dataFrame[ columnName ] <= MAX_VALUE ) & mask
	
	dataFrame = dataFrame[ mask ]
	#dataFrame.drop(labels=['kitchen_square','living_square','floor_number'], axis=1, inplace=True)
	#dataFrame = dataFrame.drop(labels=['living_square','floor_number'], axis=1, inplace=False)
	#dataFrame = dataFrame.drop(labels=['floor_number'], axis=1, inplace=False)
	if 're_id' in dataFrame.columns : dataFrame.drop(labels=['re_id'], axis=1, inplace=True)
	#if 'id' in dataFrame.columns : dataFrame.drop(labels=['id',], axis=1, inplace=True )	
	
	return dataFrame
# Neural network models	
class LinearNet(torch.nn.Module):
	def __init__(self, in_size ):
		super( LinearNet, self).__init__()
		
		self.in_size = in_size
		
		self.fc1 = torch.nn.Linear( in_size, 200); torch.nn.init.xavier_uniform_( self.fc1.weight );
		self.fc2 = torch.nn.Linear(200, 200);      torch.nn.init.xavier_uniform_( self.fc2.weight );
		self.fc3 = torch.nn.Linear(200,   1);      torch.nn.init.xavier_uniform_( self.fc3.weight );

		self.bn1 = torch.nn.BatchNorm1d(200)
		self.bn2 = torch.nn.BatchNorm1d(200)
	def forward(self, x0):
		x1 = torch.nn.functional.relu( self.bn1( self.fc1(x0) ) )
		x2 = torch.nn.functional.relu( self.bn2( self.fc2(x1) ) )
		x3 = self.fc3(x2)
		return x3
	def jacobian( self, x ):
		x0 = torch.tensor( x, requires_grad=True)
		x1 = torch.nn.functional.relu( self.bn1( self.fc1(x0) ) )
		x2 = torch.nn.functional.relu( self.bn2( self.fc2(x1) ) )
		x3 = self.fc3(x2).squeeze()
		x3.backward()
		return x0.grad
"""
# Neural network models	
class ImprovedLinearNet(torch.nn.Module):
    def __init__(self, in_size ):
        super( ImprovedLinearNet, self).__init__()
        self.in_size = in_size
        
        self.lft_fc1 = torch.nn.Linear(   2, 300); torch.nn.init.xavier_uniform_( self.lft_fc1.weight );
        self.lft_fc2 = torch.nn.Linear( 300, 300); torch.nn.init.xavier_uniform_( self.lft_fc2.weight );
        self.lft_fc3 = torch.nn.Linear( 300,   1); torch.nn.init.xavier_uniform_( self.lft_fc3.weight );

        self.lft_bn1 = torch.nn.BatchNorm1d(300)
        self.lft_bn2 = torch.nn.BatchNorm1d(300)
        
        self.rht_fc1 = torch.nn.Linear(   3, 300); torch.nn.init.xavier_uniform_( self.rht_fc1.weight );
        self.rht_fc2 = torch.nn.Linear( 300, 300); torch.nn.init.xavier_uniform_( self.rht_fc2.weight );
        self.rht_fc3 = torch.nn.Linear( 300, 1  ); torch.nn.init.xavier_uniform_( self.rht_fc3.weight );
        
        self.rht_bn1 = torch.nn.BatchNorm1d(300)
        self.rht_bn2 = torch.nn.BatchNorm1d(300)
    
    def forward(self, x0):
        #['price,total_square,longitude,latitude,number_of_rooms,number_of_floors,exploitation_start_year']
        #total_square            = x0[:, 0:1 ]
        #longitude_latitude      = x0[:,[1,2]]
        #number_of_rooms         = x0[:, 3:4 ]
        #number_of_floors        = x0[:, 4:5 ]
        #exploitation_start_year = x0[:, 5:6 ]
        
        #total_square, longitude_latitude, number_of_rooms, number_of_floors, exploitation_start_year = torch.split( x0, [1,2,1,1,1], dim=1)
        #lft_x0 = longitude_latitude
        #rht_x0 = exploitation_start_year
        total_square, lft_x0, rht_x0 = torch.split(x0, [1,2,3], dim=1 )
	
        lft_x1 = torch.nn.functional.relu( self.lft_bn1( self.lft_fc1(lft_x0) ) )
        lft_x2 = torch.nn.functional.relu( self.lft_bn2( self.lft_fc2(lft_x1) ) )
        lft_x3 = self.lft_fc3(lft_x2)
        
        rht_x1 = torch.nn.functional.relu( self.rht_bn1( self.rht_fc1(rht_x0) ) )
        rht_x2 = torch.nn.functional.relu( self.rht_bn2( self.rht_fc2(rht_x1) ) )
        rht_x3 = self.rht_fc3(rht_x2)
        
        x3 = total_square*lft_x3*rht_x3
        return x3
"""
# Neural network models	
class ImprovedLinearNet(torch.nn.Module):
    def __init__(self, in_size ):
        super( ImprovedLinearNet, self).__init__()
        self.in_size = in_size
        
        self.lft_fc1 = torch.nn.Linear(   5, 200); torch.nn.init.xavier_uniform_( self.lft_fc1.weight );
        self.lft_fc2 = torch.nn.Linear( 200, 200); torch.nn.init.xavier_uniform_( self.lft_fc2.weight );
        self.lft_fc3 = torch.nn.Linear( 200,   1); torch.nn.init.xavier_uniform_( self.lft_fc3.weight );

        self.lft_bn1 = torch.nn.BatchNorm1d(200)
        self.lft_bn2 = torch.nn.BatchNorm1d(200)
        
    def forward(self, x0):
        #['price,total_square,longitude,latitude,number_of_rooms,number_of_floors,exploitation_start_year']
        #total_square            = x0[:, 0:1 ]
        #longitude_latitude      = x0[:,[1,2]]
        #number_of_rooms         = x0[:, 3:4 ]
        #number_of_floors        = x0[:, 4:5 ]
        #exploitation_start_year = x0[:, 5:6 ]
        
        #total_square, longitude_latitude, number_of_rooms, number_of_floors, exploitation_start_year = torch.split( x0, [1,2,1,1,1], dim=1)
        #lft_x0 = longitude_latitude
        #rht_x0 = exploitation_start_year
        total_square, lft_x0 = torch.split(x0, [1,5], dim=1 )
	
        lft_x1 = torch.nn.functional.relu( self.lft_bn1( self.lft_fc1(lft_x0) ) )
        lft_x2 = torch.nn.functional.relu( self.lft_bn2( self.lft_fc2(lft_x1) ) )
        lft_x3 = self.lft_fc3(lft_x2)
        
        x3 = total_square*lft_x3
        return x3



def limitDataUsingProcentiles( dataFrame ):
	if 'price' in dataFrame.columns :
		mask = True
		
		pricePerSquare       = ( dataFrame['price']/dataFrame['total_square'] )
		pricePerSquareValues = pricePerSquare.values
		
		robustScaler = RobustScaler(quantile_range=(10, 90) )
		robustScaler.fit( pricePerSquareValues.reshape((-1,1)) )
		pricePerSquareValues = robustScaler.transform( pricePerSquareValues.reshape((-1,1)) ).reshape(-1)
		
		mask = ( pricePerSquareValues > -1 ) & ( pricePerSquareValues  < 1 ) & mask
		
		dataFrame = dataFrame[ mask ]	
	
	return dataFrame

class loadDataFrame(object) : # NUMERICAL, OBJECT, ALL
	def __call__(self, fileName, COLUMN_TYPE='NUMERICAL' ):
		dataFrame = pd.read_csv(
			fileName, 
			sep=";",
			encoding='cp1251', 
			keep_default_na=False
		).dropna(how="all")
		return self.__processDataFrame( dataFrame, COLUMN_TYPE )
	def __call__(self, dataBase, tableName, COLUMN_TYPE='NUMERICAL' ):
		engine = create_engine( dataBase )
		dataFrame = pd.read_sql_table( tableName, engine)
		return self.__processDataFrame( dataFrame, COLUMN_TYPE )
	def __processDataFrame(self, dataFrame, COLUMN_TYPE ):
		if 'price' in dataFrame.columns : dataFrame = dataFrame[ dataFrame['price'].apply( check_float ) ]
		
		dataFrame = dataFrame[ dataFrame.apply( check_row , axis=1 ) ]
		#dataFrame = dataFrame[ dataFrame.apply( check_date, axis=1 ) ]
		
		for columnName in (FLOAT_COLUMNS + INT_COLUMNS):
			if columnName in dataFrame.columns : dataFrame[ columnName ] = dataFrame[ columnName ].astype( np.float32 )
		
		#print('Shape of the data with all features:', dataFrame.shape)
		if COLUMN_TYPE == 'NUMERICAL' :
			#dataFrame = dataFrame.select_dtypes(include=['number','datetime'])
			dataFrame = dataFrame.select_dtypes(include=['number'])
		if COLUMN_TYPE == 'OBJECT'    :
			dataFrame = dataFrame.select_dtypes(include=['object'])
		
		subset = None
		if 'price' in dataFrame.columns : 
			subset=['price', 'total_square', 'number_of_rooms', 'longitude', 'latitude' ]	
		else :
			subset=['total_square', 'number_of_rooms', 'longitude', 'latitude' ]	
		dataFrame.drop_duplicates(subset=subset, keep='first', inplace=True)	
		
		return dataFrame





