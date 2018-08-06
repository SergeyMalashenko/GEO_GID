#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from commonModel import loadCSVData, FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN, QuantileRegressionLoss
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles
from commonModel import ballTreeDistance

from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors import DistanceMetric

parser = argparse.ArgumentParser()
parser.add_argument("--input"   , type=str                                            )
parser.add_argument("--output"  , type=str                                            )
parser.add_argument("--limits"  , type=str,          default=""                       )
parser.add_argument("--features", type=str, nargs=2, default=['longitude','latitude'] )

args = parser.parse_args()

inputFileName  = args.input
outputFileName = args.output
limitsFileName = args.limits
features       = args.features

inputDataFrame = loadCSVData                     ( inputFileName  )
inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsFileName )

X = inputDataFrame[ features ].values

mask = (inputDataFrame.price > 14000000) & (inputDataFrame.exploitation_start_year > 2012)
print( inputDataFrame[ mask ] )

tree = KDTree( X )
with open( outputFileName, 'wb') as fid:
	searchTreePacket = dict()
	searchTreePacket['tree'    ] = tree
	searchTreePacket['features'] = features
	searchTreePacket['data'    ] = inputDataFrame
	cPickle.dump( searchTreePacket, fid)

