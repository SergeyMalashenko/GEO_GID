# GEO_GID

### Train model
```
./trainModel.py --model modelNizhnyNovgorod.pkl --limits input/NizhnyNovgorodLimits.json --database 'mysql://database' --table 'tablename'
```
### Check model
```
Remote machine: remote_user@remote_host$ ipython notebook --no-browser --port=8889
Local  machine: local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```
Now open your browser on the local machine and type in the address bar
```
localhost:8888
```
### Test  model
```
./testModel.py --model ./modelNizhnyNovgorod.pkl --limits input/NizhnyNovgorodLimits.json --query 'latitude=56.35123,longitude=43.8699733, total_square=35, living_square=18, kitchen_square=8, number_of_rooms=1, floor_number=4, number_of_floors=9, distance_from_metro=5500, exploitation_start_year=1988'
Predicted price:  2165769.25
Predicted deltas: exploitation_start_year= 2.6538, longitude=-0.0057, latitude=-0.0038, number_of_rooms= 0.6473, total_square= 6.1633, number_of_floors= 2.2467
Predicted scales: longitude= 3.5625, latitude= 6.5746, number_of_rooms= 0.1667, total_square= 0.0022, number_of_floors= 0.0417, exploitation_start_year= 0.0118
```

### Requirements
Ubuntu 14.04/16.04, Python3 (scipy,numpy,scikit-learn,pytorch)
