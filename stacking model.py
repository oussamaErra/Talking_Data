import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import gc


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler



#------------- the stacked model: load all the predictions (6 model) and build on top of it an xgboost model-------------------------#

# load xgboost predctions with events
xg_event_train=pd.read_csv('xgb_prediction_train.csv',header=0)
xgb_event_test =pd.read_csv('xgb_predictions_test.csv',header=0)

#load xgboost predictions with no events

xgb_no_event_test=pd.read_csv('xgb_predictions_noevents_test.csv',header=0)
xgb_no_events_train=pd.read_csv('xgb_prediction_noevents_train.csv',header=0)

#load keras predictions with events : 
keras_events_test=pd.read_csv('keras_predictions_withevents_test.csv',header=0)
keras_events_train=pd.read_csv('keras_prediction_withevents_train.csv',header=0)

#load keras prediction with no events

keras_no_event_test=pd.read_csv('keras_predictions_test.csv',header=0)
keras_no_events_train=pd.read_csv('keras_prediction_train.csv',header=0)

#load extra tree end RF prediction on the data with events which also has the raget variable

rf_et_predictions_test=pd.read_csv('rf_et_predictions_test.csv',header=0)
rf_et_prediction_train=pd.read_csv('rf_et_prediction_train.csv',header=0)
columns = rf_et_prediction_train.columns.tolist()
Target_name = columns[len(columns)-1]
target= rf_et_prediction_train[Target_name]
rf_et_prediction_train.drop(Target_name,axis=1,inplace=True)

# concat all train and test data in the same ordere 

train=pd.concat((xg_event_train,xgb_no_events_train,keras_events_train,keras_no_events_train,rf_et_prediction_train), axis = 1)
test=pd.concat((xgb_event_test,xgb_no_event_test,keras_events_test,keras_no_event_test,rf_et_predictions_test),axis=1)

#now i have the train data ready , test data ready ant the target variable , lets build our final Xgboost model

################################## Actual Run Code ##################################

lable_group = LabelEncoder()
Y = lable_group.fit_transform(target)

# enter the number of folds from xgb.cv
ntest=test.shape[0]
folds = 5
early_stopping = 50
oof_test = np.zeros((ntest,12))



start_time = timer(None)

# Load data set and target values


d_test = xgb.DMatrix(test)

# set up KFold that matches xgb.cv number of folds
kf = StratifiedKFold(target, n_folds=folds,random_state=0)

#Start the CV
for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = train.values[train_index], train.values[test_index]
    y_train, y_val = Y[train_index], Y[test_index]

#######################################
#
# Define cross-validation variables
#
#######################################

    params = {}
    params["booster"]= "gbtree"
    params['objective'] = "multi:softprob"
    params['eval_metric'] ='mlogloss'
    params['num_class']=12
    params['eta'] = 0.01
    params['gamma'] = 0.1
    params['min_child_weight'] = 1
    params['colsample_bytree'] = 0.5
    params['subsample'] = 0.8
    params['max_depth'] = 7  
    params['silent'] = 1
    params['random_state'] = 0

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

####################################
#  Build Model
####################################

################################
#PS : if you would like to test the performance of the model please make sure to give a notice to early stoping ,for some versions of xgboost it maximize the mlogloss not minimize
################################
    clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping) 
    timer(start_time)
####################################
#  Evaluate Model and Predict
####################################

    oof_test[:] += clf.predict(d_test, ntree_limit=clf.best_iteration)  
    print(' eval-log_loss: %.6f' % log_loss( y_val, clf.predict(d_valid), ntree_limit=clf.best_iteration))
   

####################################
#Average predictions 
####################################

  

oof_test /= folds



####################################
#  Make a submision
####################################

result = pd.DataFrame(oof_test, columns=lable_group.classes_)
test=pd.read_csv('C:/Users/oussama/Documents/first/gender_age_test.csv',header=0)
result["device_id"] = test.device_id
result = result.set_index("device_id")
now = datetime.now()
sub_file = 'Errabia_Oussama_submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
result.to_csv(sub_file, index=True,index_label='device_id')