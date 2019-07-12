#!/usr/bin/env python

import argparse
import json
import os

from flask      import Flask, Response, request
from sqlalchemy import create_engine

from getSimilarObjects import getSimilarObjectsMain

ENV_HOST = os.getenv('MYSQL_HOST'     , default = '127.0.0.1')
ENV_USER = os.getenv('MYSQL_ROOT_USER', default = 'root'     )
ENV_PASS = os.getenv('MYSQL_ROOT_PASSWORD')

app = Flask(__name__)

engine = None 
def parseArguments() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--database",                    type=str)
    parser.add_argument("--host"    , default='0.0.0.0', type=str)
    parser.add_argument("--port"    , default=5000     , type=int)
    parser.add_argument("--verbose" , action="store_true")
    
    args = parser.parse_args()
    
    databaseName = args.database
    appHost      = args.host
    appPort      = args.port 
    verboseFlag  = args.verbose
    
    return databaseName, appHost, appPort, verboseFlag 

@app.route('/api/users/<getSimilarObjectsRequest>')
def getSimilarObjectsRequest():
    tableName  = str ()
    userQuery  = dict()
    userScales = dict()
    userDeltas = dict()
    outputTopK     = 0
    outputFeatures = list()
    
    tableName      = request.args.get('tableName')
    userQuery      = request.args.get('userQuery' )
    userScales     = request.args.get('userScales')
    userDeltas     = request.args.get('userDeltas')
    outputTopK     = request.args.get('outputTopK')
    outputFeatures = request.args.get('outputFeatures')
    
    closestItem_s = getSimilarObjectsMain( engine, tableName, userQuery, userScales, userDeltas, outputTopK, outputFeatures )
    json_output = closestItem_s.to_dict( orient='records' )
    resp = Response( json.dumps( json_output, default=json_serial, sort_keys=True, indent=4, separators=(',', ': ')) )
    resp.headers['Content-Type'] = 'application/json'
    return resp
if __name__ == "__main__":
    databaseName, appHost, appPort, verboseFlag = parseArguments()
    engine = create_engine( databaseName )
    app.run( host=appHost, port=appPort )


