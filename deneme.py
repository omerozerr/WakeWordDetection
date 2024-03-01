####### IMPORTS #############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")

####### Making our data training-ready
X = df["feature"].values
print(X.shape)