"""
https://www.kaggle.com/code/cdeotte/xgboost-5000-mutations-200-pdb-files-lb-0-410

"""

#Importing libraries
import os
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from utilities.metricUtils import AccuracyComputer

######################### Importing dataset ####################################
PATH="hypotheses/ML-results/"

if not os.path.exists(PATH): os.makedirs(PATH)

##--------------------- Feature Engineered -------------------------------------
train = pd.read_csv('datasets/feature_pdb/train-embedding.csv')
validate = pd.read_csv('datasets/feature_pdb/holdout-embedding.csv') #Validation
test = pd.read_csv('datasets/feature_pdb/test-embedding.csv')

FEATURES = ['position', 'relative_position', 'Molecular Weight_1', 'Molecular Weight_2', 'Molecular Weight_delta', 'Residue Weight_1', 'Residue Weight_2', 'Residue Weight_delta', \
'pKa1_1', 'pKa1_2', 'pKa1_delta', 'pKb2_1', 'pKb2_2', 'pKb2_delta',  'pl4_1', 'pl4_2', 'pl4_delta', 'H_1', 'H_2', 'H_delta', 'VSC_1', 'VSC_2', \
'VSC_delta', 'P1_1', 'P1_2', 'P1_delta', 'P2_1', 'P2_2', 'P2_delta', 'SASA_1', 'SASA_2', 'SASA_delta', 'NCISC_1', 'NCISC_2', 'NCISC_delta', 'blosum100', 'blosum80', 'blosum60',\
 'blosum40', 'demask', 'cos_angle', 'location3d', 'pca_pool_0', 'pca_wt_0', 'pca_mutant_0', 'pca_local_0', 'pca_pool_1', 'pca_wt_1', 'pca_mutant_1', 'pca_local_1', \
'pca_pool_2', 'pca_wt_2', 'pca_mutant_2', 'pca_local_2', 'pca_pool_3', 'pca_wt_3', 'pca_mutant_3', 'pca_local_3', 'pca_pool_4', 'pca_wt_4', 'pca_mutant_4', \
'pca_local_4', 'pca_pool_5', 'pca_wt_5', 'pca_mutant_5', 'pca_local_5', 'pca_pool_6', 'pca_wt_6', 'pca_mutant_6', 'pca_local_6', 'pca_pool_7', 'pca_wt_7', \
'pca_mutant_7', 'pca_local_7', 'pca_pool_8', 'pca_wt_8', 'pca_mutant_8', 'pca_local_8', 'pca_pool_9', 'pca_wt_9', 'pca_mutant_9', 'pca_local_9', 'pca_pool_10', \
'pca_wt_10', 'pca_mutant_10', 'pca_local_10', 'pca_pool_11', 'pca_wt_11', 'pca_mutant_11', 'pca_local_11', 'pca_pool_12', 'pca_wt_12', 'pca_mutant_12', \
'pca_local_12', 'pca_pool_13', 'pca_wt_13', 'pca_mutant_13', 'pca_local_13', 'pca_pool_14', 'pca_wt_14', 'pca_mutant_14', 'pca_local_14', 'pca_pool_15', \
'pca_wt_15', 'pca_mutant_15', 'pca_local_15', 'pca_pool_16', 'pca_pool_17', 'pca_pool_18', 'pca_pool_19', 'pca_pool_20', 'pca_pool_21', 'pca_pool_22',\
 'pca_pool_23', 'pca_pool_24', 'pca_pool_25', 'pca_pool_26', 'pca_pool_27', 'pca_pool_28', 'pca_pool_29', 'pca_pool_30', 'pca_pool_31', 'mut_prob',\
 'mut_entropy', 'sa_total', 'sa_apolar', 'sa_backbone', 'sa_sidechain', 'sa_ratio', 'sa_in/out', 'AA1', 'AA2', 'AA3', 'AA4']
X_train = train.loc[:, FEATURES]
y_train = train.loc[:,'target']
X_valid = validate.loc[:, FEATURES]
y_valid = validate.loc[:, 'target']
X_test = test.loc[:, FEATURES]
sid_test = test['seq_id']

##------------------------ Embedding based -------------------------------------

# def load_hdf5_to_numpy(h5file, csvfile, mode='train'):
#     h5_obj = h5py.File(h5file)
#     keys = h5_obj.keys()
#     xarr = np.zeros((len(keys), 1024))
#     if mode !='infer':
#         datadf = pd.read_csv(csvfile)
#         yarr = np.zeros((len(keys))) # tm values
#     else:
#         yarr = [] # Seq_id for inference submission
#     for i, k in enumerate(keys):
#         xarr[i][:] = h5_obj[k][:]
#         if mode !='infer':
#             yarr[i]    = datadf.loc[datadf['seq_id'] == int(k)]['tm']
#         else:
#             yarr.append(int(k))

#     h5_obj.close()
#     return xarr, yarr

# X_train, y_train = load_hdf5_to_numpy('datasets/Nesp-Train-bioembed.hdf5',
#                                 'datasets/train_split.csv')
# X_valid, y_valid = load_hdf5_to_numpy('datasets/Nesp-Valid-bioembed.hdf5',
#                                 'datasets/valid_split.csv')
# X_test, sid_test = load_hdf5_to_numpy('datasets/Nesp-Test-bioembed.hdf5',
#                                 'datasets/test.csv', mode='infer')

##################### Model Config ######################################
TITLE = "RF-Results"

param_grid = {
    "n_estimators": [ 500 ],
    "min_samples_leaf": [3],
    "min_samples_split": [3],
    "criterion": ["squared_error"],
 }
regressor = RandomForestRegressor()

# param_grid = [
#     # {"kernel": ['rbf'], 'gamma': ['auto', 'scale']}
#     # {"kernel": ['sigmoid'], 'gamma': ['auto', 'scale']}
#     # {"kernel": ['poly'], 'gamma': ['auto', 'scale'], 'degree': [3,5,9]}
# ]

# regressor = SVR(max_iter =10000)


grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)


######################### Results & Logging ####################################

model = grid_search.best_estimator_
y_pred = model.predict(X_valid)

err = AccuracyComputer()
err.prd = y_pred
err.tgt = y_valid
print(grid_search.best_params_)
print(grid_search.best_params_, file=open(PATH+TITLE+"_log.txt", 'a'))
print(err.print_summary(ret=True))
print(err.print_summary(ret=True), file=open(PATH+TITLE+"_log.txt", 'a'))

submit = pd.DataFrame()
submit['true_tm'] = y_valid
submit['pred_tm'] = y_pred
submit.to_csv(PATH+TITLE+'valid-pred.csv', index=False)

###----------------- Create Kaggle Test Submission --------------------
sub_pred = model.predict(X_test)
submit = pd.DataFrame(sid_test, columns=['seq_id'])
submit['tm'] = sub_pred
submit.to_csv(PATH+TITLE+'_submission.csv', index=False)

