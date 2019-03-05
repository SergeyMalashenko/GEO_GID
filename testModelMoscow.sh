#export DATABASE_='mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev' 
#export TABLE_='nn_neiro'
export DATABASE_='mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock' 
export TABLE_='real_estate_from_ads_api' 
export MODELNAME_=modelMoscow.pkl 
export MODE_="--verbose"
#export MODE_=""

set -x
#https://www.avito.ru/moskva/kvartiry/1-k_kvartira_30_m_25_et._998335776 4750000
./testModel.py --model $MODELNAME_ --query 'latitude=55.658268, longitude=37.3548943, total_square=30, living_square=16, kitchen_square=7, number_of_rooms=1, floor_number=2, number_of_floors=5, exploitation_start_year=1974, distance_to_metro=1600' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/2-k_kvartira_38_m_112_et._1596482271 6100000
./testModel.py --model $MODELNAME_ --query 'latitude=55.716443, longitude=37.7287053, total_square=38, living_square=23, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=12, exploitation_start_year=1969, distance_to_metro=900' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/3-k_kvartira_87_m_28_et._987440216 22500000
./testModel.py --model $MODELNAME_ --query 'latitude=55.741419, longitude=37.5383314, total_square=87, living_square=55, kitchen_square=11, number_of_rooms=3, floor_number=2, number_of_floors=8, exploitation_start_year=1937, distance_to_metro=400' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/4-k_kvartira_147_m_717_et._1231798133 43500000
./testModel.py --model $MODELNAME_ --query 'latitude=55.793479, longitude=37.4847623, total_square=147, living_square=98, kitchen_square=18, number_of_rooms=4, floor_number=7, number_of_floors=17, exploitation_start_year=1999, distance_to_metro=400' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/2-k_kvartira_39_m_412_et._1124607490 8000000
./testModel.py --model $MODELNAME_ --query 'latitude=55.809714, longitude=37.5607683, total_square=39, living_square=23, kitchen_square=7, number_of_rooms=2, floor_number=4, number_of_floors=12, exploitation_start_year=1967, distance_to_metro=1100' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/1-k_kvartira_19.7_m_25_et._1259496315 3940000
./testModel.py --model $MODELNAME_ --query 'latitude=55.708173, longitude=37.6797551, total_square=20, living_square=17, kitchen_square=5, number_of_rooms=1, floor_number=2, number_of_floors=5, exploitation_start_year=1951, distance_to_metro=400' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/2-k_kvartira_44_m_39_et._1468514090 6350000
./testModel.py --model $MODELNAME_ --query 'latitude=55.707575, longitude=37.8184922, total_square=44, living_square=30, kitchen_square=7, number_of_rooms=2, floor_number=3, number_of_floors=9, exploitation_start_year=1969, distance_to_metro=900' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/3-k_kvartira_78_m_1617_et._1047715382 18000000
./testModel.py --model $MODELNAME_ --query 'latitude=55.7790048, longitude=37.4527849, total_square=78, living_square=50, kitchen_square=10, number_of_rooms=3, floor_number=16, number_of_floors=17, exploitation_start_year=1997, distance_to_metro=2800' --limits input/MoscowLimits.json
#https://www.avito.ru/moskva/kvartiry/3-k_kvartira_97.6_m_712_et._1303570760 47000000
./testModel.py --model $MODELNAME_ --query 'latitude=55.7751889, longitude=37.5901012, total_square=97.6, living_square=50, kitchen_square=10, number_of_rooms=3, floor_number=7, number_of_floors=12, exploitation_start_year=1987, distance_to_metro=400' --limits input/MoscowLimits.json

set +x
