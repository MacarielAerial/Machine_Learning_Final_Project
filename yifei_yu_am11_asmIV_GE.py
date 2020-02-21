from __future__ import absolute_import, division, print_function

# Import libraries
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, SimpleRNN, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import os

# Configure the programme
plt.style.use('seaborn-darkgrid')
random.seed(11)

# Specify global parameters
input_path = 'GE Stock Price with LSTM/'
input_file = 'ge.us.txt'
output_path = '/output/'
output_file = 'output.csv'
features = ['open', 'volume', 'close']
time_col_name = 'day'
n_steps = 4
n_steps_out = 2
batch_size = 10
rnn_unit = False
more_layer = True

class Solver:
	'''Executes sub-modules through this class object'''	
	def __init__(self, input_path, input_file, output_path, output_file, features, time_col_name, n_steps, batch_size, rnn_unit, more_layer, train_test_split = 0.85, n_steps_out = 0):
		'''Loads global variables into the object'''
		Aux.directory_create(input_path + output_path[1:-1]) # Create a directory if not already exist
		self.input_path = input_path
		self.input_file = input_file
		self.output_path = output_path
		self.output_file = output_file
		self.features = features
		self.time_col_name = time_col_name
		self.n_steps = n_steps
		self.batch_size = batch_size
		self.rnn_unit = rnn_unit # Specify whether to enable RNN as a substitute for LSTM
		self.more_layer = more_layer # Specify whether to add another layer for the model
		self.train_test_split = train_test_split
		self.n_steps_out = n_steps_out # Specify the number of timesteps for predictions
		self.n_features = len(self.features)
		self.df = Aux.data_import(self.input_path + self.input_file)
		self.train = pd.DataFrame()
		self.val = pd.DataFrame()
		self.test = pd.DataFrame()
		self.X_train = pd.DataFrame()
		self.y_train = pd.DataFrame()
		self.X_val = pd.DataFrame()
		self.y_val = pd.DataFrame()
		self.X_test = pd.DataFrame()
		self.y_test = pd.DataFrame()
		self.model1 = None
		self.model2 = None
		self.model3 = None
		self.scaler_train = StandardScaler() # Intitiate a set of scalers
		self.scaler_val = StandardScaler()
		self.scaler_test = StandardScaler()
		self.predictions = None
		self.test_scroes = None

	def diagnostics(self):
		'''Displays basic configuration info'''
		Aux.diagnostics()

	def data_clean(self):
		'''Cleans the data'''
		self.df = Aux.column_names_clean(self.df) # Tidy up column names
		self.df[self.time_col_name] = pd.to_datetime(self.df[self.time_col_name]) # Set datetime column
		#self.df.set_index(time_col_name, inplace = True)
		self.df['close'] = np.sqrt(self.df['close']) # Square root transform target feature column
		self.df = Aux.column_reorder(self.df) # Reorder columns so target feature column is at the end
		self.df = self.df[self.features]

	def data_vis(self):
		'''Create visualisations of raw data'''
		plt.figure(figsize = (6,6))
		self.df['close'].plot()
		plt.savefig(self.input_path + self.output_path + 'plot_1.png', dpi = 300)
		plt.close()

	def data_split(self):
		'''Splits data into train and test sets without shuffling'''
		self.train, self.test = np.split(self.df, [int(self.train_test_split * len(self.df))])
		self.train, self.val = np.split(self.train, [int(self.train_test_split * len(self.train))]) # Split data into three folds
		self.train, self.val, self.test = self.scaler_train.fit_transform(self.train), self.scaler_val.fit_transform(self.val), self.scaler_test.fit_transform(self.test) # Scale three sets of data separately
		self.X_train, self.y_train = Aux.data_reshape(self.train, self.n_steps, self.n_steps_out)
		self.X_val, self.y_val = Aux.data_reshape(self.val, self.n_steps, self.n_steps_out)
		self.X_test, self.y_test = Aux.data_reshape(self.test, self.n_steps, self.n_steps_out) # Reshape data to the correct dimensionality for LSTM models

	def model_define(self):
		'''Defines the model architecture'''
		self.model1 = Sequential()
		self.model1.add(BatchNormalization(input_shape = (self.n_steps, self.n_features)))
		self.model1.add(Dropout(0.2))
		if self.rnn_unit:
			if self.more_layer:
				self.model1.add(SimpleRNN(64, activation = 'relu', return_sequences = True))
				self.model1.add(BatchNormalization())
				self.model1.add(SimpleRNN(16, activation = 'relu'))
			else:
				self.model1.add(SimpleRNN(64, activation = 'relu'))
		else:
			if self.more_layer:
				self.model1.add(LSTM(64, activation = 'relu', return_sequences = True))
				self.model1.add(BatchNormalization())
				self.model1.add(LSTM(16, activation = 'relu'))
			else:
				self.model1.add(LSTM(64, activation = 'relu'))
		if self.n_steps_out > 0:
			self.model1.add(Dense(self.n_steps_out))
		else:
			self.model1.add(Dense(1)) # Enables addition of layers and substitutes of computational units with a series of if statements
		self.model1.summary() # Print out a model architecture summary
		plot_model(self.model1, self.input_path + self.output_path + 'model.png', show_shapes = True) # Save a computational graph
		self.model1.save(self.input_path + self.output_path + 'model') # Save the model for future usage
		self.model1.compile(loss = keras.losses.logcosh, # Compile the model with set parameters
				    optimizer = keras.optimizers.Adam(),
				    metrics = ['mse'])

	def model_train(self):
		'''Trains the model with historical data'''
		history = self.model1.fit(self.X_train, self.y_train,
					 batch_size = self.batch_size,
					 epochs = 100,
					 validation_data = (self.X_val, self.y_val),
					 callbacks = Aux.earlystopping(),
					 verbose = 0)
		Aux.train_history_vis(history.history, path = self.input_path + self.output_path) # Visualise training history
	
	def model_eval(self):
		'''Evaluates model performance on test data'''
		# Predicts test data with trained model
		self.predictions = self.model1.predict(self.X_test)
		# Only take the first timestep prediction/actual if multistep forecast is used
		if self.n_steps_out > 0:
			self.predictions = np.array([i[0] for i in self.predictions])
			self.y_test = np.array([i[0] for i in self.y_test])
		# Evaluates model performance with Tensorflow's built in function
		self.test_scores = self.model1.evaluate(self.X_test, self.y_test, verbose = 2)
		print('Test loss:', self.test_scores[0])
		print(self.predictions)
		# De-normalises data for plotting
		self.predictions, self.y_test = Aux.denormalisation(self.predictions, self.y_test, self.scaler_test)
		# Visualises de-normalised actual vs. predicted data
		Aux.actual_vs_prediction_vis(self.predictions, self.y_test, self.input_path + self.output_path)

	def exec(self):
		self.diagnostics()
		self.data_clean()
		self.data_vis()
		self.data_split()
		self.model_define()
		self.model_train()
		self.model_eval()

class Aux:
	'''Supports the main solver module'''
	def data_import(path):
		'''Imports data'''
		df = pd.read_csv(path, sep = ',')
		return df

	def column_names_clean(df):
		'''Cleans column names'''
		df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
		return df

	def directory_create(path):
		'''Create a directory if not already exists'''
		if not os.path.exists(path):
			os.makedirs(path)

	def diagnostics():
		print('*'*10 + 'Tensorflow Version' + '*'*10)
		print('*'*10 + tf.__version__ + '*'*10)

	def earlystopping():
		'''Returns a Keras earlystopping object'''
		return [EarlyStopping(monitor = 'mse', patience = 10)]

	def column_reorder(df):
		'''Reorder dataframe columns so the target feature is at the end'''
		column_labels = df.columns.values.tolist()
		target_label = column_labels.pop(-3)
		column_labels.append(target_label)
		return df[column_labels]

	def denormalisation(predictions, y_test, scaler_test):
		'''De-normalises data that was centred and scaled for training purpose'''
		# Zero padding for de-normalisation
		predictions = np.c_[np.zeros((len(predictions), 2)), predictions]
		# De-normalisation
		predictions = scaler_test.inverse_transform(predictions)[:, -1]
		# Reverses square root transformation
		predictions = np.square(predictions)
		# Zero padding for de-normalisation
		y_test = np.c_[np.zeros((len(y_test), 2)), y_test]
		# De-normalisation
		y_test = scaler_test.inverse_transform(y_test)[:, -1]
		# Reverse square root transformation
		y_test = np.square(y_test)
		return predictions, y_test

	def data_reshape(df, n_steps, n_steps_out):
		'''Splits multiple parallel time-series into three-dimensional samples for LSTM'''
		X, y = list(), list()
		for i in range(len(df)):
			# Find the end of this sample as well as the end of forecast period if required
			end_ix = i + n_steps
			out_end_ix = end_ix + n_steps_out
			# Check if we have gone beyond the data
			if end_ix >= len(df):
				break
			elif out_end_ix > len(df):
				break
			# Gather input and output parts of the sample
			if n_steps_out > 0:
				seg_x, seg_y = df[i:end_ix, :], df[end_ix:out_end_ix, -1]
			else:
				seg_x, seg_y = df[i:end_ix, :], df[end_ix, -1]
			X.append(seg_x)
			y.append(seg_y)
		return np.array(X), np.array(y)

	def train_history_vis(history, path):
		'''Visualises the model's training history in terms of loss and accuracy'''
		# Plot training & validation loss values
		plt.figure(figsize = (16,9))
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc = 'upper left')
		plt.savefig(path + 'Loss History.png', dpi = 300)
		plt.close()

	def actual_vs_prediction_vis(predictions, y_test, path):
		'''Visualises the actual test data against the model's predictions'''
		plt.figure(figsize = (16,9))
		plt.plot(y_test, color = 'blue')
		plt.plot(predictions, color = 'red')
		plt.title('Test Actuals vs. Predictions')
		plt.xlabel('Date')
		plt.ylabel('GE Stock Price')
		plt.legend(['Actuals', 'Predictions'], loc = 'upper left')
		plt.savefig(path + 'Actuals Predictions' + str(random.randint(1, 100)) + '.png', dpi = 300)
		plt.close()

		

def main():
	solver_object = Solver(input_path = input_path, input_file = input_file, output_path = output_path, output_file = output_file, features = features, time_col_name = time_col_name, n_steps = n_steps, batch_size = batch_size, rnn_unit = rnn_unit, more_layer = more_layer, n_steps_out = n_steps_out)
	solver_object.exec()

if __name__ == '__main__':
	main()
