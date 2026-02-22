import numpy as np
import tensorflow as tf
from config import RANDOM_SEED

def set_seed():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

def print_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")