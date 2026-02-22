SEQ_LEN = 60
TRAIN_SPLIT = 0.7
RANDOM_SEED = 42

FEATURES = [
    'max_temp','min_temp','avg_temp',
    'humidity','rainfall','wind_speed',
    'pressure','cloud_cover'
]

TARGET_INDEX = [2, 4]  # avg_temp, rainfall
CITY = "Hyderabad"