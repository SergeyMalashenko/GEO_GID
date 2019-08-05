#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest
from sklearn.neighbors         import LocalOutlierFactor

from sklearn.model_selection   import train_test_split
from sklearn.model_selection   import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble          import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression

from sklearn.preprocessing     import QuantileTransformer
from sklearn.preprocessing     import LabelEncoder
from sklearn.preprocessing     import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.tree              import export_graphviz

from sklearn.pipeline          import Pipeline

from scipy.spatial.distance    import mahalanobis

import math

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import pydot

import torch
import torch.optim

from commonModel import loadDataFrame, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN, QuantileRegressionLoss, HuberRegressionLoss
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles
from commonModel import ImprovedLinearNet, LinearNet

import matplotlib
#matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--input"   , type=str, default="" )

parser.add_argument("--database", type=str, default="" )
parser.add_argument("--table"   , type=str, default="" )

parser.add_argument("--model"  , type=str, default="" )
parser.add_argument("--seed"   , type=int, default=43 )
parser.add_argument("--output" , type=str, default="" )
parser.add_argument("--limits" , type=str, default="" )

parser.add_argument("--features", type=str            )
parser.add_argument("--verbose" , action="store_true" )

args = parser.parse_args()

def preProcessData( dataFrame, targetColumn, seed ):
	"""
	def excludeAnomaliesIsolationForest( dataFrame, targetColumn ):
		Y_data = dataFrame    [[ targetColumn ]];       Y_values = Y_data.values;
		X_data = dataFrame.drop( targetColumn, axis=1); X_values = X_data.values;
		clf = IsolationForest(); clf.fit( X_values )
		
		y_noano = clf.predict( X_values )
		y_noano = pd.DataFrame(y_noano, columns = ['Top'])
		y_noano[y_noano['Top'] == 1].index.values
		
		dataFrame = dataFrame.iloc[y_noano[y_noano['Top'] == 1].index.values]
		print("IsolationForest algorithm" )
		print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
		print("Number of rows without outliers:", dataFrame.shape[0])
		return dataFrame
	"""
	def excludeAnomaliesIsolationForest( dataFrame ):
		processedDataFrame = pd.DataFrame(index=dataFrame.index)
		processedDataFrame['price_square'           ] = dataFrame['price']/dataFrame['total_square']
		processedDataFrame['longitude'              ] = dataFrame['longitude']
		processedDataFrame['latitude'               ] = dataFrame['latitude' ]
		processedDataFrame['exploitation_start_year'] = dataFrame['exploitation_start_year']
		
		X_values = processedDataFrame.values;
		preprocessor = MinMaxScaler()
		preprocessor.fit( X_values )
		X_values = preprocessor.transform( X_values )
	
		clf     = IsolationForest(); clf.fit( X_values );
		y_noano = clf.predict( X_values )
		y_noano = pd.DataFrame(y_noano, columns = ['Top'])
		y_noano[y_noano['Top'] == 1].index.values
		
		print( dataFrame.shape[0] )
		dataFrame = dataFrame.iloc[y_noano[y_noano['Top'] == 1].index.values]
		print("IsolationForest algorithm")
		print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
		print("Number of rows without outliers:", dataFrame.shape[0])
		
		return dataFrame
	
	def excludeAnomaliesLocalOutlierFactor( dataFrame ):
		processedDataFrame = pd.DataFrame(index=dataFrame.index)
		processedDataFrame['price_square'           ] = dataFrame['price']/dataFrame['total_square']
		processedDataFrame['longitude'              ] = dataFrame['longitude']
		processedDataFrame['latitude'               ] = dataFrame['latitude' ]
		processedDataFrame['exploitation_start_year'] = dataFrame['exploitation_start_year']
		
		X_values = processedDataFrame.values;
		preprocessor = MinMaxScaler()
		preprocessor.fit( X_values )
		X_values = preprocessor.transform( X_values )
	
		clf     = LocalOutlierFactor( n_neighbors=20 )
		y_noano = clf.fit_predict( X_values )
		y_noano = pd.DataFrame(y_noano, columns = ['Top'])
		y_noano[y_noano['Top'] == 1].index.values
		
		print( dataFrame.shape[0] )
		dataFrame = dataFrame.iloc[y_noano[y_noano['Top'] == 1].index.values]
		print("LocalOutlierFactor algorithm")
		print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
		print("Number of rows without outliers:", dataFrame.shape[0])
		
		return dataFrame
	
	def selectFeatures( dataFrame, targetColumn ):
		index  = dataFrame.index
		Y_data = dataFrame    [[ targetColumn ]]
		X_data = dataFrame.drop( targetColumn, axis=1)
		
		X_values = X_data.values
		Y_values = Y_data.values
		
		selection = SelectKBest  ( f_regression, k='all' )
		selector  = selection.fit( X_values, Y_values )
		
		oldFeatureNames = list( X_data.columns.values ); newFeatureNames = []
		scores = selection.scores_
		mask   = selection.get_support()
		for bool, score, featureName in zip(mask, scores, oldFeatureNames ):
			if bool:
				print( "{:17} {}".format( featureName, score ) )
				newFeatureNames.append( featureName )
		X_values_ = selector.transform( X_values )
		Y_values_ = Y_values
		
		dataFrame                 = pd.DataFrame( X_values_, index=index, columns=newFeatureNames, dtype=np.float64 )
		dataFrame[ targetColumn ] = Y_values_
		return dataFrame
	
	dataFrame = excludeAnomaliesIsolationForest   ( dataFrame )
	dataFrame = excludeAnomaliesLocalOutlierFactor( dataFrame )
	#dataFrame = selectFeatures                    ( dataFrame, targetColumn )
	
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.width', 175 ):
		print( dataFrame.describe() )
	
	return dataFrame
"""
def postProcessData( INDEX_test, X_test, Y_test, Y_predict ) :
	threshold_s = [2.5, 5.0, 10.0, 15.0 ]
	
	Y_predict =  np.array( Y_predict )
	Y_test    =  np.array( Y_test    )
	Y_rel_err = (np.abs( Y_predict - Y_test )/Y_test*100 ).astype( np.int )
	
	for threshold in threshold_s : 
		bad_s  = np.sum( ( Y_rel_err  > threshold ).astype( np.int ) )
		good_s = np.sum( ( Y_rel_err <= threshold ).astype( np.int ) )
		print("threshold = {:5}%, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, bad_s/(good_s+bad_s)) )
	
	mask       = Y_rel_err > 25
	INDEX_test = INDEX_test[ mask ]
	X_test     = X_test    [ mask ]
	Y_test     = Y_test    [ mask ]
	Y_predict  = Y_predict [ mask ]
	
	index_s    = np.argsort( Y_test )
	INDEX_test = INDEX_test[ index_s ]
	X_test     = X_test    [ index_s ]
	Y_test     = Y_test    [ index_s ]
	Y_predict  = Y_predict [ index_s ]
	
	for i in range( INDEX_test.size ) :
		index     = INDEX_test[ i ]+2
		x_test    = X_test    [ i ]
		y_test    = Y_test    [ i ]
		y_predict = Y_predict [ i ] 
		print('{:6} {:10.1f} {:10.1f} {:10.1f}%'.format( index, y_test, y_predict, (y_predict-y_test)*100./y_test ))
"""

def trainNeuralNetworkModel( dataFrame, targetColumn, featureNames, seed=43 ):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    dataFrame = dataFrame[ featureNames ]
    	
    FEATURE_NAMES = list( dataFrame.columns ); FEATURE_NAMES.remove( targetColumn )
    COLUMNS       = list( dataFrame.columns );
    LABEL         = targetColumn;
    	
    Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
    X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
    Y_values    = Y_values
    	
    print( X_dataFrame.describe() )
    
    FEATURE_DEFAULTS = ((X_dataFrame.max()+X_dataFrame.min())*0.5).to_dict()
    	
    #preprocessorY = MinMaxScaler()
    #preprocessorY = StandardScaler()
    preprocessorY = MaxAbsScaler()
    preprocessorY.fit( Y_values )
    preprocessorX = MinMaxScaler()
    #preprocessorX = StandardScaler()
    preprocessorX.fit( X_values )
    	
    Y_values = preprocessorY.transform( Y_values )
    X_values = preprocessorX.transform( X_values )
    	
    device = torch.device('cpu')
    #device = torch.device('cuda') # Uncomment this to run on GPU
    
    #Create model
    in_size = len( FEATURE_NAMES )
    #model = ConvolutionalNet( in_size ).to( device )
    model = LinearNet( in_size ).to( device )
    #model = ImprovedLinearNet( in_size ).to( device )
    	
    learning_rate = 0.01
    #loss_fn       = torch.nn.SmoothL1Loss()
    #loss_fn       = QuantileRegressionLoss( 0.5 ) 
    #loss_fn       = HuberRegressionLoss( 0.15 ) 
    #loss_fn       = torch.nn.MSELoss  ( size_average=False)
    loss_fn       = torch.nn.L1Loss  ( )
    #optimizer     = torch.optim.SGD   ( model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer     = torch.optim.Adam  ( model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0.001 )
    scheduler     = torch.optim.lr_scheduler.StepLR( optimizer, step_size=250, gamma=0.5)
    
    batch_size           = 256
    average_nbr_corrects = 0; N = 100; alpha = 2./(N+1); 
    current_nbr_corrects = 0;
    	
    X_numpyTrainVal, X_numpyTest, Y_numpyTrainVal, Y_numpyTest = train_test_split( X_values, Y_values, test_size=0.1 )
    X_torchTest   = torch.from_numpy( X_numpyTest.astype( np.float32 ) ).to( device )
    Y_torchTest   = torch.from_numpy( Y_numpyTest.astype( np.float32 ) ).to( device )
    X_torchTest_s = torch.split( X_torchTest, batch_size, dim=0 )
    Y_torchTest_s = torch.split( Y_torchTest, batch_size, dim=0 )
    	
    for t in range(6000):
        model.train()
        X_numpyTrain, X_numpyVal, Y_numpyTrain, Y_numpyVal = train_test_split( X_numpyTrainVal, Y_numpyTrainVal, test_size=0.25 )
	
        X_torchTrain = torch.from_numpy( X_numpyTrain.astype( np.float32 ) ).to( device )
        X_torchVal   = torch.from_numpy( X_numpyVal  .astype( np.float32 ) ).to( device )
        Y_torchTrain = torch.from_numpy( Y_numpyTrain.astype( np.float32 ) ).to( device )
        Y_torchVal   = torch.from_numpy( Y_numpyVal  .astype( np.float32 ) ).to( device )
        	
        train_size     = X_numpyTrain.shape[0]
        val_size       = X_numpyVal  .shape[0]
        	
        train_index_s  = torch.randperm( train_size )
        X_torchTrain_s = X_torchTrain[ train_index_s ]
        Y_torchTrain_s = Y_torchTrain[ train_index_s ]
        val_index_s    = torch.randperm( val_size )
        X_torchVal_s   = X_torchVal  [ val_index_s ]
        Y_torchVal_s   = Y_torchVal  [ val_index_s ]
        	
        X_torchTrain_s = torch.split( X_torchTrain, batch_size, dim=0 )
        Y_torchTrain_s = torch.split( Y_torchTrain, batch_size, dim=0 )
        X_torchVal_s   = torch.split( X_torchVal  , batch_size, dim=0 )
        Y_torchVal_s   = torch.split( Y_torchVal  , batch_size, dim=0 )
        
        length = ( len( X_torchVal_s ) - 1)*batch_size
        #Train
        for i in range( len(Y_torchTrain_s)-1 ):
            x = X_torchTrain_s[i]
            y = Y_torchTrain_s[i]
            
            y_pred = model(x)
            loss   = loss_fn( (y_pred - y)/y , torch.zeros( y.shape ) )
            		
            model.zero_grad()
            loss.backward  ()
            optimizer.step ()
        scheduler.step()
        #Validate
        model.eval()
	
        ValLoss = 0
        Y_torchPredict   = torch.zeros( Y_torchVal.shape, dtype=torch.float32 ).to( device )
        Y_torchPredict_s = torch.split( Y_torchPredict, batch_size, dim=0 )
        for i in range( len(Y_torchPredict_s)-1 ):
            x      = X_torchVal_s[i]
            y      = Y_torchVal_s[i]
            y_pred = model(x) 
            
            Y_torchPredict_s[i].copy_( y_pred )
            ValLoss += loss_fn( y_pred, y )
        ValLoss /= (len(Y_torchPredict_s)-1) 
        Y_numpyPredict = Y_torchPredict.cpu().detach().numpy()
        
        threshold = 0.1; eps = 0.001
        ValTrue_s   = np.sum( np.abs( (Y_numpyPredict - Y_numpyVal)/( Y_numpyVal + eps ) ) <= threshold )
        ValFalse_s  = np.sum( np.abs( (Y_numpyPredict - Y_numpyVal)/( Y_numpyVal + eps ) ) >  threshold )
        ValAccuracy = float(ValTrue_s)/(ValTrue_s + ValFalse_s)
         
        TestLoss = 0; TestAccuracy = 0;
        if Y_torchTest.nelement() > 0 :
            model.eval()
            TestLoss = 0
            Y_torchPredict   = torch.zeros( Y_torchTest.shape, dtype=torch.float32 ).to( device )
            Y_torchPredict_s = torch.split( Y_torchPredict, batch_size, dim=0 )
            for i in range( len(Y_torchPredict_s)-1 ):
                x      = X_torchTest_s[i]
                y      = Y_torchTest_s[i]
                y_pred = model(x) 
                
                Y_torchPredict_s[i].copy_( y_pred )
                TestLoss += loss_fn( y_pred, y )
            TestLoss /= (len(Y_torchPredict_s)-1) 
            Y_numpyPredict = Y_torchPredict.cpu().detach().numpy()
            
            threshold = 0.1; eps = 0.001
            TestTrue_s   = np.sum( np.abs( (Y_numpyPredict - Y_numpyTest)/( Y_numpyTest + eps ) ) <= threshold )
            TestFalse_s  = np.sum( np.abs( (Y_numpyPredict - Y_numpyTest)/( Y_numpyTest + eps ) ) >  threshold )
            TestAccuracy = float(TestTrue_s)/(TestTrue_s + TestFalse_s)
        
        print( "epoch: {:6d}, lr: {:8.6f}, val_loss: {:6.4f}, val_acc: {:6.4f}, test_loss: {:6.4f}, test_acc: {:6.4f}".format( t, get_lr(optimizer), ValLoss, ValAccuracy, TestLoss, TestAccuracy ) )
        
    # Check model
    model.eval()
    
    X_numpyTotal = X_values
    Y_numpyTotal = Y_values
    
    X_torchTotal = torch.from_numpy( X_numpyTotal.astype( np.float32 ) ).to( device )
    Y_torchTotal = torch.from_numpy( Y_numpyTotal.astype( np.float32 ) ).to( device )
    Y_torchPredict = model( X_torchTotal )
    Y_numpyPredict = Y_torchPredict.cpu().detach().numpy()
    Y_numpyTotal   = Y_torchTotal  .cpu().detach().numpy()
    
    eps = 0.001
    Y_relErr = np.abs( Y_numpyPredict - Y_numpyTotal )/( Y_numpyTotal + eps )
    for threshold in [ 0.025, 0.05, 0.10, 0.15 ]:
        bad_s   = np.sum( ( Y_relErr  > threshold ) )
        good_s  = np.sum( ( Y_relErr <= threshold ) )
        total_s = Y_relErr.size
        print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format( threshold, good_s, bad_s, good_s/(good_s+bad_s)) )
    
    Y_numpyPredict = preprocessorY.inverse_transform( Y_numpyPredict )
    Y_numpyTotal   = preprocessorY.inverse_transform( Y_numpyTotal   )
    
    modelPacket = dict()
    modelPacket['model'           ] = model
    modelPacket['preprocessorX'   ] = preprocessorX
    modelPacket['preprocessorY'   ] = preprocessorY
    
    modelPacket['feature_names'   ] = FEATURE_NAMES
    modelPacket['feature_defaults'] = FEATURE_DEFAULTS
    
    return modelPacket, ( Y_numpyPredict, Y_numpyTotal )

def postProcessData( modelPacket, dataFrame, targetColumn, featureNames ) :
    dataFrame = dataFrame[ featureNames ]
    
    FEATURE_NAMES = list( dataFrame.columns ); FEATURE_NAMES.remove( targetColumn )
    COLUMNS       = list( dataFrame.columns );
    LABEL         = targetColumn;
    
    Y_dataFrame = dataFrame    [[ targetColumn ]];       Y_values = Y_dataFrame.values;
    X_dataFrame = dataFrame.drop( targetColumn, axis=1); X_values = X_dataFrame.values;
    Y_values    = Y_values
    	
    preprocessorY = modelPacket['preprocessorY']
    preprocessorY.fit( Y_values )
    preprocessorX = modelPacket['preprocessorX']
    preprocessorX.fit( X_values )
    
    Y_values = preprocessorY.transform( Y_values )
    X_values = preprocessorX.transform( X_values )
    
    X_numpyTotal = X_values
    Y_numpyTotal = Y_values
    
    device = torch.device('cpu')
    #device = torch.device('cuda') # Uncomment this to run on GPU
    
    X_torchTotal = torch.from_numpy( X_numpyTotal.astype( np.float32 ) ).to( device )
    Y_torchTotal = torch.from_numpy( Y_numpyTotal.astype( np.float32 ) ).to( device )
    
    Y_torchPredict = modelPacket['model']( X_torchTotal )
    Y_numpyPredict = Y_torchPredict.detach().numpy()
    Y_numpyTotal   = Y_torchTotal  .detach().numpy()
    
    Y_numpyPredict = preprocessorY.inverse_transform( Y_numpyPredict )
    Y_numpyTotal   = preprocessorY.inverse_transform( Y_numpyTotal   )
    
    threshold = 10
    
    Y_relativeError = np.abs( Y_numpyPredict - Y_numpyTotal )*100/Y_numpyTotal
    pricePerSquare  = ( dataFrame.price / dataFrame.total_square ).values.reshape(-1,1)
    
    allValues = pricePerSquare
    mask = Y_relativeError > threshold
    badValues  = pricePerSquare[ mask ] 
    mask = Y_relativeError <= threshold
    goodValues = pricePerSquare[ mask ]
    
    bins = range(5,25)
    bins = [i * 0.5e4 for i in bins]
    	
    figure, axes =plt.subplots(3,1)
    axes[1].axis('tight')
    axes[1].axis('off')
    
    resultValues = axes[0].hist([ allValues, goodValues, badValues ], bins=bins, histtype='bar', color=['green','yellow','red'])
    allValues  = resultValues[0][0]; goodValues = resultValues[0][1]; badValues = resultValues[0][2];
    
    accuracy = goodValues*100/(allValues+0.01)
    col_label = ['{:5d}'.format( int((bins[i+0]+bins[i+1])/2) ) for i in range( len(bins)-1 ) ]
    cell_text = [['{:2.1f}'.format( acc_ ) for acc_ in accuracy],]
    
    table_ = axes[1].table(cellText=cell_text, colLabels=col_label,loc='center')
    table_.auto_set_font_size(False)
    table_.set_fontsize(8)
    
    Y_numpyTotal_max    = np.max( Y_numpyTotal )
    Y_numpyTotal_min    = np.min( Y_numpyTotal )
    Y_numpyTotal_width  = Y_numpyTotal_max-Y_numpyTotal_min
    Y_numpyTotal_height = Y_numpyTotal_max-Y_numpyTotal_min
   
    #axes[2].set_position([Y_numpyTotal_min-Y_numpyTotal_width*0.1,Y_numpyTotal_min-Y_numpyTotal_width*0.1,Y_numpyTotal_width*0.2,Y_numpyTotal_width*0.2])
    axes[2].plot   ( Y_numpyTotal, Y_numpyTotal, c='blue' )
    axes[2].plot   ( Y_numpyTotal, Y_numpyTotal*(1.0 + 0.1), c='red'  )
    axes[2].plot   ( Y_numpyTotal, Y_numpyTotal*(1.0 - 0.1), c='red'  )
    axes[2].scatter( Y_numpyPredict, Y_numpyTotal )
    plt.show()
    
    #figure, axes =plt.subplots(3,1)
    #clust_data = np.random.random((10,3))
    #collabel=("col 1", "col 2", "col 3")
    #axs[0].axis('tight')
    #axs[0].axis('off')
    #the_table = axs[0].table(cellText=clust_data,colLabels=collabel,loc='center')
    	
    #axs[1].plot(clust_data[:,0],clust_data[:,1])
    #plt.show()
    	
    return
 
inputFileName = args.input    #NizhniyNovgorod.csv
inputDatabase = args.database #mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev nn
inputTable    = args.table    #nn

modelFileName  = args.model
outputFileName = args.output
limitsFileName = args.limits
seed           = args.seed

featureNames   = (args.features).split(',')
verboseFlag    = args.verbose

trainDataFrame = None
if inputDatabase != "" and inputTable != "" : 
	trainDataFrame = loadDataFrame()( inputDatabase, inputTable )
if verboseFlag :
    print( trainDataFrame.describe() )

#trainDataFrame = limitDataUsingLimitsFromFilename( trainDataFrame, limitsFileName )
trainDataFrame = trainDataFrame.select_dtypes(include=['number'])
#trainDataFrame = limitDataUsingProcentiles       ( trainDataFrame )

if verboseFlag :
    print( trainDataFrame.describe() )

#trainDataFrame = preProcessData( trainDataFrame, TARGET_COLUMN, seed )
trainedModelPacket, ( Y_predict, Y_test ) = trainNeuralNetworkModel   ( trainDataFrame, TARGET_COLUMN, featureNames, seed )
postProcessData( trainedModelPacket, trainDataFrame, TARGET_COLUMN, featureNames )

if modelFileName != "" :
	with open( modelFileName, 'wb') as fid:
		cPickle.dump( trainedModelPacket, fid)
