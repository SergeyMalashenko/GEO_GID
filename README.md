# GEO_GID

Train model
```
./trainModel.py --input input/NizhnyNovgorod.csv --model model.pkl

```
Check model
```
./checkModel.py --input input/NizhnyNovgorod_train.csv --model model.pkl
```
Test  model
```
./testModel.py --model model.pkl --input input/NizhnyNovgorod_test.csv
./testModel.py --model model.pkl --query 'longitude=44.075417, latitude=56.283864, total_square=43.0, living_square=14.0, kitchen_square=11.0, number_of_rooms=1, floor_number=9, number_of_floors=17'
```
