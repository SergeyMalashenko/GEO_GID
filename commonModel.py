import pandas            as pd
import numpy             as np

FLOAT_COLUMNS = [ 'price', 'longitude', 'latitude']
INT_COLUMNS   = [ 'total_square', 'living_square', 'kitchen_square', 'number_of_rooms', 'floor_number', 'number_of_floors' ]
STR_COLUMNS   = [ 'type', 'bulding_type' ]
TARGET_COLUMN =   'price'

MIN_PRICE            = 100000; MAX_PRICE            = 100000000;
MIN_TOTAL_SQUARE     = 12    ; MAX_TOTAL_SQUARE     = 500      ;
MIN_LIVING_SQUARE    = 8     ; MAX_LIVING_SQUARE    = 300      ;
MIN_KITCHEN_SQUARE   = 4     ; MAX_KITCHEN_SQUARE   = 100      ;
MIN_NUMBER_OF_ROOMS  = 1     ; MAX_NUMBER_OF_ROOMS  = 10       ;
MIN_FLOOR_NUMBER     = 1     ; MAX_FLOOR_NUMBER     = 50       ;
MIN_NUMBER_OF_FLOORS = 1     ; MAX_NUMBER_OF_FLOORS = 50       ;
MIN_LATITUDE         = 56.10 ; MAX_LATITUDE         = 56.50    ;
MIN_LONGITUDE        = 43.70 ; MAX_LONGITUDE        = 44.30    ;

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
	def preprocessing( dataFrame ) :
		mask = True	
		mask = (dataFrame['price'          ] > MIN_PRICE          ) & (dataFrame['price'          ] < MAX_PRICE          ) & mask
		mask = (dataFrame['total_square'   ] > MIN_TOTAL_SQUARE   ) & (dataFrame['total_square'   ] < MAX_TOTAL_SQUARE   ) & mask
		mask = (dataFrame['longitude'      ] > MIN_LONGITUDE      ) & (dataFrame['longitude'      ] < MAX_LONGITUDE      ) & mask
		mask = (dataFrame['latitude'       ] > MIN_LATITUDE       ) & (dataFrame['latitude'       ] < MAX_LATITUDE       ) & mask
		mask = (dataFrame['living_square'  ] > MIN_LIVING_SQUARE  ) & (dataFrame['living_square'  ] < MAX_LIVING_SQUARE  ) & mask
		mask = (dataFrame['kitchen_square' ] > MIN_KITCHEN_SQUARE ) & (dataFrame['kitchen_square' ] < MAX_KITCHEN_SQUARE ) & mask
		mask = (dataFrame['number_of_rooms'] > MIN_NUMBER_OF_ROOMS) & (dataFrame['number_of_rooms'] < MAX_NUMBER_OF_ROOMS) & mask
		
		mask = (dataFrame['floor_number'   ] > MIN_FLOOR_NUMBER   ) & (dataFrame['floor_number'   ] < MAX_FLOOR_NUMBER   ) & mask
		
		dataFrame = dataFrame[ mask ]	
		
		return dataFrame
	
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
	dataFrame = preprocessing( dataFrame )
	print('Shape of the data with numerical features:', dataFrame.shape)
	
	return dataFrame
