# AE_based_anomaly_detection.py

# The following code is based on examples which are found in the repository [1] 
# of SINDy-augmented AE [2]. Additions and modifications to accommodate the 
# anomaly detection application are made.
# [1] https://github.com/kpchamp/SindyAutoencoders
# [2] K. Champion, B. Lusch, J. N. Kutz, and S. L. Brunton, "Data-driven 
# discovery of coordinates and governing equations," Proceedings of the National 
# Academy of Sciences, 116 (2019), pp. 22445–22451.

!git clone https://github.com/kpchamp/SindyAutoencoders.git
import sys
sys.path.append('/content/SindyAutoencoders/src')
import os
import pandas as pd
import numpy as np
from sindy_utils import library_size
from training import train_network
import tensorflow as tf
import pynumdiff.total_variation_regularization as tvr
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import pickle
from autoencoder import full_network
from training import create_feed_dictionary

mat = loadmat('measurement_time_series.mat')
meas_ts = mat['meas_ts']
list_corrupted_segments = mat['list_corrupted_segments']

# For IEEE 14 bus system, we have 32 rows in measurement Jacobian matrix.

data = np.transpose(meas_ts.reshape((32,-1)))

# For IEEE 30 bus system, we have 100 rows in measurement Jacobian matrix.

# data = np.transpose(meas_ts.reshape((100,-1)))

# 250 pertains to the number of time steps encompassed within each segment.

segments = np.split(data, data.shape[0]/250)

# -1 is necessitated in list_corrupted_segments.reshape(-1)-1, 
# since MATLAB indices start at 1.
list_healthy_segments = list(np.arange(0,data.shape[0]/250))
for i in (list_corrupted_segments.reshape(-1)-1):
  list_healthy_segments.remove(i)

segments_healthy = [segments[i] for i in list_healthy_segments]

# Computation of measurement derivatives based on total variation regularization.
dXdt = []
for i in range(0,len(segments)):
    T = segments[i].shape[0]
    dXdt_hat = np.zeros(T)
    for j in range(0,segments[i].shape[1]):
        _, dxdt_hat = tvr.iterative_velocity(segments[i][:,j], dt=0.02, params=[1,0.001])
        dXdt_hat = np.vstack((dXdt_hat, dxdt_hat))
    dXdt_hat = np.delete(dXdt_hat, 0, axis=0)
    dXdt.append(dXdt_hat.T)

segments_healthy_dXdt = [dXdt[i] for i in list_healthy_segments]

segments_ = {}
segments_['x'] = segments_healthy
segments_['dx'] = segments_healthy_dXdt

df_data = DataFrame.from_dict(segments_)

# Healthy measurement segments are eventually split in 3 subsets.

X_train_test, X_out_of_sample = train_test_split(df_data, test_size=0.2, shuffle=True)

out_of_sample_data = {}
out_of_sample_data['x'] = X_out_of_sample.x.to_list()
out_of_sample_data['dx'] = X_out_of_sample.dx.to_list()

X_train, X_test = train_test_split(X_train_test, test_size=0.2, shuffle=True)

train_data = {}
train_data['x'] = X_train.x.to_list()
train_data['dx'] = X_train.dx.to_list()

test_data = {}
test_data['x'] = X_test.x.to_list()
test_data['dx'] = X_test.dx.to_list()

# X_train, X_test, and X_out_of_sample are redefined as dictionaries.

X_train = {}
X_test = {}
X_out_of_sample = {}

# For IEEE 14 bus system:

X_train['x'] = np.asarray(train_data['x']).reshape((-1,32))
X_test['x'] = np.asarray(test_data['x']).reshape((-1,32))
X_out_of_sample['x'] = np.asarray(out_of_sample_data['x']).reshape((-1,32))
X_train['dx'] = np.asarray(train_data['dx']).reshape((-1,32))
X_test['dx'] = np.asarray(test_data['dx']).reshape((-1,32))
X_out_of_sample['dx'] = np.asarray(out_of_sample_data['dx']).reshape((-1,32))

# For IEEE 30 bus system:

# X_train['x'] = np.asarray(train_data['x']).reshape((-1,100))
# X_test['x'] = np.asarray(test_data['x']).reshape((-1,100))
# X_out_of_sample['x'] = np.asarray(out_of_sample_data['x']).reshape((-1,100))
# X_train['dx'] = np.asarray(train_data['dx']).reshape((-1,100))
# X_test['dx'] = np.asarray(test_data['dx']).reshape((-1,100))
# X_out_of_sample['dx'] = np.asarray(out_of_sample_data['dx']).reshape((-1,100))

# Set of hyper-paramaeters associated with SINDy-augmented AE, where the meaning 
# of each setting is described in https://github.com/kpchamp/SindyAutoencoders

params = {}

params['input_dim'] = X_train['x'].shape[1]
params['latent_dim'] = 4
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# Sequential thresholding parameters.

params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# Weights of conventional AE loss function.

params['loss_weight_decoder'] = 1.0

# Contribution of SINDy-related terms in the AE objective function is controlled 
# by adjusting the following three hyper-parameters. Fine tuning of SINDy-augmented 
# AE hyper-parameters, including among others the following three weights, was 
# prohibitive given computational constraints.
# For plain AE in the absence of inducing physical context in the latent space, 
# these three objective function weights are set to 0.

params['loss_weight_sindy_z'] = 1e-5
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'

# For IEEE 14 bus system measurement configutation:

params['widths'] = [16,8]

# For IEEE 30 bus system measurement configutation:

# params['widths'] = [50,25]

params['epoch_size'] = X_train['x'].shape[0]

# 250 pertains to the number of time steps involved in a measurement segment. Thus, 
# batch size is set at this number.

params['batch_size'] = 250 

params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 1

params['max_epochs'] = 10000
params['refinement_epochs'] = 1000

# Depending on the objective function weights associated with SINDy, as set in 
# params['loss_weight_sindy_z'], params['loss_weight_sindy_x'], and 
# params['loss_weight_sindy_regularization'], training of AE or SINDy-augmented AE  
# is conducted.

df = pd.DataFrame()
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['save_name'] = 'experiment'
tf.reset_default_graph()
results_dict = train_network(X_train, X_test, params)
df = df.append({**results_dict, **params}, ignore_index=True)
df.to_pickle('experiment_results' + '.pkl')

data_path = os.getcwd() + '/'
save_name = 'experiment'
params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
params['save_name'] = data_path + save_name
tf.reset_default_graph()
autoencoder_network = full_network(params)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

tensorflow_run_tuple = ()
for key in autoencoder_network.keys():
    tensorflow_run_tuple += (autoencoder_network[key],)
    
# Computation of maximum reconstruction error based on segments utilized during 
# training phase.

reconstruction_error = []
for j in range(0,len(segments_['x'])):
  examine_segment = {}
  examine_segment['x'] = segments_['x'][j]
  examine_segment['dx'] = segments_['dx'][j]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, data_path + save_name)
    test_dictionary = create_feed_dictionary(examine_segment, params)
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

  results = {}
  for i,key in enumerate(autoencoder_network.keys()):
    results[key] = tf_results[i]

  reconstruction_error.append(np.linalg.norm(examine_segment['x'] - results['x_decode']))

max_reconstruction_error = np.max(np.asarray(reconstruction_error))

# Out-of-sample data for computation of evaluation metrics consist of the 
# previous test set pertaining to out-of-sample healthy segments, augmented by 
# the corrupted segments which were initially set aside.

segments_corrupted = [segments[i] for i in (list_corrupted_segments.reshape(-1)-1)]
segments_corrupted_dXdt = [dXdt[i] for i in (list_corrupted_segments.reshape(-1)-1)]

test_segments = {}

# Lists consisting of the aforementioned measurement segments are merged.

test_segments['x'] = np.split(X_out_of_sample['x'],X_out_of_sample['x'].shape[0]/250) + segments_corrupted 
test_segments['dx'] = np.split(X_out_of_sample['x'],X_out_of_sample['x'].shape[0]/250) + segments_corrupted_dXdt
list_out_of_sample_healthy_segments = list(np.arange(0,len(np.split(X_out_of_sample['x'],X_out_of_sample['x'].shape[0]/250))))
list_out_of_sample_corrupted_segments = list(np.arange(len(np.split(X_out_of_sample['x'],X_out_of_sample['x'].shape[0]/250)),len(test_segments['x'])))

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
for j in range(0,len(test_segments['x'])):
  test_segment = {}
  test_segment['x'] = test_segments['x'][j]
  test_segment['dx'] = test_segments['dx'][j]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, data_path + save_name)
    test_dictionary = create_feed_dictionary(test_segment, params)
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

  results = {}
  for i,key in enumerate(autoencoder_network.keys()):
    results[key] = tf_results[i]

  reconstruction_error = np.linalg.norm(test_segment['x'] - results['x_decode'])

  if (reconstruction_error > max_reconstruction_error):
    if j in list_out_of_sample_corrupted_segments:
      true_positives += 1
    else:
      false_positives += 1
  else:
    if j in list_out_of_sample_healthy_segments:
      true_negatives += 1
    else:
      false_negatives += 1

total_cases = len(test_segments['x'])

accuracy = (true_positives + true_negatives)/total_cases
precision = true_positives/(true_positives + false_positives)
sensitivity = true_positives/(true_positives + false_negatives)
f1_score = 2*precision*sensitivity/(precision+sensitivity)    