# GEO_GID

Train model
```
./trainModel.py --input input/NizhnyNovgorodWithYear.csv --model NizhnyNovgorodModelPacket.pkl --limits input/NizhnyNovgorodLimits.json
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

Build dataset with 

Test  model
```
./buildSearchTree.py --input ./input/NizhnyNovgorod.csv --limits ./input/NizhnyNovgorodLimits.json --output ./input/searchTreeNizhnyNovgorod.pkl
./testModel.py --model NizhnyNovgorodModelPacket.pkl ---limits input/NizhnyNovgorodLimits.json --query 'longitude=44.075417, latitude=56.283864, total_square=43.0, living_square=14.0, kitchen_square=11.0, number_of_rooms=1, floor_number=9, number_of_floors=17' [--dataset input/NizhnyNovgorod.csv --tolerances 'longitude=0.001, latitude=0.001, total_square=5']

```
