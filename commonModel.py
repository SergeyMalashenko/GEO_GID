import pandas            as pd
import numpy             as np

FLOAT_COLUMNS = [ 'price', 'longitude', 'latitude']
INT_COLUMNS   = [ 'total_square', 'living_square', 'kitchen_square', 'number_of_rooms', 'floor_number', 'number_of_floors' ]
STR_COLUMNS   = [ 'type', 'bulding_type' ]
TARGET_COLUMN =   'price'

def check_float( x ):
	try:
		float(x)
	except ValueError:
		return False
	return True

def check_row( row ):
	check_float_s = check_float( row.price ) and check_float( row.longitude ) and check_float( row.latitude )
	return check_float_s

def loadData( fileName ):
	dataFrame = pd.read_csv(
		fileName, 
		sep=";",
		encoding='cp1251', 
		verbose=True, 
		keep_default_na=False
	).dropna(how="all")
	
	dataFrame = dataFrame[ dataFrame.apply( check_row, axis=1 ) ]
	
	dataFrame['price'    ] = dataFrame['price'    ].astype(np.float64)
	dataFrame['longitude'] = dataFrame['longitude'].astype(np.float64)
	dataFrame['latitude' ] = dataFrame['latitude' ].astype(np.float64)	
	
	print('Shape of the data with all features:', dataFrame.shape)
	dataFrame = dataFrame.select_dtypes(exclude=['object'])
	print('Shape of the data with numerical features:', dataFrame.shape)
	print("List of features contained our dataset:",list( dataFrame.columns ))
	
	dataFrame.drop_duplicates(subset=['price', 'total_square', 'number_of_rooms', 'longitude', 'latitude' ], keep='first', inplace=True)	
	print('Shape of the data with numerical features:', dataFrame.shape)
	
	mask = True	
	mask = (dataFrame['price'          ] > 1.0*1e6) & (dataFrame['price'       ] < 6.0*1e6 ) & mask
	mask = (dataFrame['total_square'   ] > 20     ) & (dataFrame['total_square'] < 200     ) & mask
	mask = (dataFrame['longitude'      ] > 0      ) & mask
	mask = (dataFrame['latitude'       ] > 0      ) & mask
	mask = (dataFrame['living_square'  ] > 10     ) & mask
	mask = (dataFrame['kitchen_square' ] > 4      ) & mask
	mask = (dataFrame['number_of_rooms'] > 0      ) & mask
	
	dataFrame = dataFrame[ mask ]	
	
		
	return dataFrame


