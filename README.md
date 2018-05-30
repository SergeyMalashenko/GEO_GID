# GEO_GID

Train model
```
./trainModel.py --input input/NizhnyNovgorod.csv --model model.pkl

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
./testModel.py --model model.pkl --input input/NizhnyNovgorod_test.csv
./testModel.py --model model.pkl --query 'longitude=44.075417, latitude=56.283864, total_square=43.0, living_square=14.0, kitchen_square=11.0, number_of_rooms=1, floor_number=9, number_of_floors=17'
```
