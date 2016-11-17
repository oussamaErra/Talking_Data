#gxboost model on data without events


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
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

def create_submission(score, prediction):
    # Make Submission
    test=pd.read_csv('C:/Users/oussama/Documents/first/gender_age_test.csv',header=0)
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

DATA_TRAIN_PATH = 'C:/Users/oussama/Documents/TalkingData/gender_age_train.csv'
DATA_TEST_PATH = 'C:/Users/oussama/Documents/TalkingData/gender_age_test.csv'
BRAND_PATH='C:/Users/oussama/Documents/TalkingData/phone_brand_device_model.csv'

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH,path_brand_phne=BRAND_PATH):
    # Phone brand

    brand=pd.read_csv(path_brand_phne, dtype={'device_id': np.str})
    brand.drop_duplicates('device_id', keep='first', inplace=True)
    brand['phone_brand'] = pd.factorize( brand['phone_brand'],sort=True)[0]
    brand['device_model'] = pd.factorize( brand['device_model'],sort=True)[0]   
    
    train_loader = pd.read_csv(path_train, dtype={'device_id': np.str})
    train = train_loader.drop(['age', 'gender'], axis=1)
    #train['group'] = pd.factorize( train['group'],sort=True)[0] 
    train = pd.merge(train, brand, how='left', on='device_id', left_index=True)
    # target
    target=train.group
   
    train.drop(['device_id','group'],axis =1 , inplace = True)
    train.fillna(-1, inplace=True)
    #train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)


    test_loader = pd.read_csv(path_test, dtype={'device_id': np.str})
    test = pd.merge(test_loader, brand, how='left', on='device_id', left_index=True)
    test.drop('device_id',axis =1 , inplace = True)
    test.fillna(-1, inplace=True)

 

    return train,test,target

################################## Actual Run Code ##################################



train , test , y_train = load_data()
gc.collect()
lable_group = LabelEncoder()
Y = lable_group.fit_transform(y_train)

NFOLDS = 5
SEED = 0



print("{},{}".format(train.shape, test.shape))


x_train = train.values
ntrain=train.shape[0]
x_test = test.values
ntest=test.shape[0]

kf =StratifiedKFold(y_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)


params = {
    "objective": "multi:softprob",
    'min_child_weight': 1,
    "num_class": 12,
    "booster": "gbtree",
    'colsample_bytree': 0.5,  
    'subsample': 0.8,
    "max_depth": 4,
    "eval_metric": "mlogloss",
    "eta": 0.01,
    "silent": 1,
    "alpha": 1,
    'gamma': 0,
    'seed': SEED
    }
oof_train = np.zeros((ntrain,12))
oof_test = np.zeros((ntest,12))

for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = x_train[train_index], x_train[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    ################################
        #PS : if you would like to test the performance of the model please make sure to give a notice to early stoping ,for some versions of xgboost it maximize the mlogloss not minimize
    ################################
    clf = xgb.train(params,
                    d_train,
                    10000,
                    evals= watchlist ,early_stopping_rounds=60)

    oof_test[:] += clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_iteration)
    oof_train[test_index]=clf.predict(xgb.DMatrix( X_val), ntree_limit=clf.best_iteration)
    
    
oof_test /= NFOLDS



xgb_predictions_test = pd.DataFrame(oof_test)
xgb_prediction_train = pd.DataFrame(oof_train)

xgb_predictions_test.to_csv('xgb_predictions_noevents_test.csv',index=None)
xgb_prediction_train.to_csv('xgb_prediction_noevents_train.csv',index=None)


print('-------- next stup : Stacking -----------')