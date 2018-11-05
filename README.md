# GEO_GID

Train model
```
./trainModel.py --model modelNizhnyNovgorod.pkl --limits input/NizhnyNovgorodLimits.json --database 'mysql://database' --table 'tablename'
```
Check model
```
Remote machine: remote_user@remote_host$ ipython notebook --no-browser --port=8889
Local  machine: local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```
Now open your browser on the local machine and type in the address bar
```
localhost:8888
```
Test  model
```
./testModel.py --model modelNizhnyNovgorod.pkl --query 'longitude=44.0730544, latitude=56.286023, total_square=35, living_square=18, kitchen_square=9, number_of_rooms=1, floor_number=2, number_of_floors=10, exploitation_start_year=1991' --limits input/NizhnyNovgorodLimits.json --database 'mysql://database' --table 'tablename' --tolerances 'longitude=0.001, latitude=0.001'
Predicted value 2,493,668
Median value    2,128,654
Mean value      2,188,734
Max value       2,916,666
Min value       1,720,155
[{"re_id":1249},...,{"re_id":109169}]
```
