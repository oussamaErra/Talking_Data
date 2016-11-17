 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import gc
    




DATA_TRAIN_PATH = 'C:/Users/oussama/Documents/TalkingData/gender_age_train.csv'
DATA_TEST_PATH = 'C:/Users/oussama/Documents/TalkingData/gender_age_test.csv'
BRAND_PATH='C:/Users/oussama/Documents/TalkingData/phone_brand_device_model.csv'
EVENTS_PATH='C:/Users/oussama/Documents/TalkingData/events.csv'
APP_EVENTS_PATH='C:/Users/oussama/Documents/TalkingData/app_events.csv'
APP_LABELS_PATH='C:/Users/oussama/Documents/TalkingData/app_labels.csv'
LABELS_CATEGORY='C:/Users/oussama/Documents/TalkingData/label_categories.csv'

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



def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH,path_brand_phne=BRAND_PATH ,events_path=EVENTS_PATH ,app_events_path=APP_EVENTS_PATH,app_labels_path= APP_LABELS_PATH,labels_category_path=LABELS_CATEGORY):
    # Phone brand

    brand=pd.read_csv(path_brand_phne)
    brand.drop_duplicates('device_id', keep='first', inplace=True)
    brand['phone_brand'] = pd.factorize( brand['phone_brand'],sort=True)[0]
    # add new feature : the number of occurence (popularity) of each phne brand
    dict_phone=dict(brand.phone_brand.value_counts())
    brand['phone_brand_occurence'] =  brand.phone_brand.apply(dict_phone.get)

    
    brand['device_model'] = pd.factorize( brand['device_model'],sort=True)[0]   
    # add new feature : the number of occurence (popularity) of each phne brand
    dict_device=dict(brand.device_model.value_counts())
    brand['device_model_occurence'] = brand.device_model.apply(dict_device.get)

    train_loader = pd.read_csv(path_train)
    train = train_loader.drop(['age', 'gender'], axis=1)
    train['group'] = pd.factorize( train['group'],sort=True)[0] 
    train = pd.merge(train, brand, how='left', on='device_id', left_index=True)
    gc.collect()
    
    #installed app 
    appevents = pd.read_csv(app_events_path,header=0,nrows = 10000)
    events = pd.read_csv(events_path,header=0)
    appencoder = LabelEncoder().fit(appevents.app_id)
    appevents['app'] = appencoder.transform(appevents.app_id)
    instaled_app= pd.merge(appevents , events.loc[:,['event_id','device_id']] , how='left' ,on ='event_id' )
    #instaled_app.reset_index(inplace=True)
    gc.collect()
    #label data
    applabels = pd.read_csv(app_labels_path)
    labels_ctegory=pd.read_csv(labels_category_path)

    #factorize the labels category
    labels_ctegory.category = pd.factorize(labels_ctegory.category)[0]

    #merge the app labels and labels category
    applabels=pd.merge(applabels ,labels_ctegory , on ='label_id' , how = 'left')
    gc.collect()
    
    #add new features: the number of occurence (popularity ) of each label id 
    dict_label=dict(applabels.label_id.value_counts())
    applabels['label-occurence'] = applabels.label_id.apply(dict_label.get)
    gc.collect()
    #select only those in application events 
    applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
   
    #transforme the app_id with the encoder already defined 
    applabels['app'] = appencoder.transform(applabels.app_id)
    
    # perform a one hot encoding for the label id  
    labelencoder = LabelEncoder().fit(applabels.label_id)
    applabels['label'] = labelencoder.transform(applabels.label_id)
    gc.collect()
    #merge the installed_app and the applabels so by then to merge them with the train data 
    label_features=pd.merge(instaled_app.loc[:,['device_id','app']],applabels.loc[:,['app','label','label-occurence','category']] , on ='app' , how ='left')
    #label_features.reset_index(inplace=True)
    




    # merge the train data with new set to mark For each device which apps it has installed

    #train = pd.merge(train , instaled_app , how = 'left',on='device_id')
    
    # merge the train data with label_features to mark For each device the label of the app used.
    
    train = pd.merge(train , label_features,how ='left' , on ='device_id')
    train.fillna(-1, inplace=True)
    
    #add features:
    # 1 : #number of times of the same app used per device:
    frame =pd.DataFrame( train.loc[:,['device_id','app']].groupby(['device_id','app']).size() , columns=['nbr_same_app']).reset_index()
    gc.collect()
    train=pd.merge(train,frame , on= ['device_id','app'] , how ='left' )
    #rectify the 1 in number of times the same app  for devices with no events
    train.loc[train.app ==-1.0,'nbr_same_app'] =-1

    # 2 : #number of times of the same label used per device:
    frame =pd.DataFrame( train.loc[:,['device_id','label']].groupby(['device_id','label']).size() , columns=['nbr_same_label']).reset_index()
    gc.collect()
    train=pd.merge(train,frame , on= ['device_id','label'] , how ='left' )
    #rectify the 1 in number of times the same app  for devices with no events
    train.loc[train.app ==-1.0,'nbr_same_label'] =-1

    # 3 : #number of times of the same category used per device:
    frame =pd.DataFrame( train.loc[:,['device_id','category']].groupby(['device_id','category']).size() , columns=['nbr_same_category']).reset_index()
    gc.collect()
    train=pd.merge(train,frame , on= ['device_id','category'] , how ='left')
    #rectify the 1 in the number of same app  for devices with no events
    train.loc[train.app ==-1.0,'nbr_same_category'] =-1

    #4 : number of times of the same occurence of each device 
    dict_device = dict(train.loc[:,['device_id','app']].groupby(['device_id'])['app'].agg(np.size))
    train['device_occur'] = train.device_id.apply(dict_device.get)
    # 5 sum of labels
    dict_device = dict(train.loc[:,['device_id','nbr_same_label']].groupby(['device_id'])['nbr_same_label'].agg(np.sum))
    train['sum_of_labels'] = train.device_id.apply(dict_device.get)

    #6 sum of app
    dict_device = dict(train.loc[:,['device_id','nbr_same_app']].groupby(['device_id'])['nbr_same_app'].agg(np.sum))
    train['sum_of_app'] = train.device_id.apply(dict_device.get)
    
    #7 sum of category
    dict_device = dict(train.loc[:,['device_id','nbr_same_category']].groupby(['device_id'])['nbr_same_category'].agg(np.sum))
    train['sum_of_category'] = train.device_id.apply(dict_device.get)

    
    #done from adding new features , drop the deplicated device_ids
    train.drop_duplicates('device_id' , keep ='first' , inplace =True )
    train.reset_index(drop=True , inplace=True)
    train.loc[train.app!=-1,['app','label','category']]=1
    train.drop(['label-occurence','nbr_same_app','nbr_same_label','nbr_same_category'],axis =1 , inplace = True)


   
    # target
    target=train.group
    

    #drop the device id and target from the train data 
    train.drop(['device_id','group'],axis =1 , inplace = True)
    


    test_loader = pd.read_csv(path_test)
    test = pd.merge(test_loader, brand, how='left', on='device_id', left_index=True)
    # merge the test data with new set to mark For each device which apps it has installed

    #test = pd.merge(test , instaled_app , how = 'left',on='device_id')
    # merge the train data with label_features to mark For each device the label of the app used.
    test = pd.merge(test, label_features,how ='left' , on ='device_id')
    
    
    
    test.fillna(-1, inplace=True)
    
    # the same for the test set , we add the previous features , i proceded in this manner (each one , test and train , alone) for the fact of memory
    #add features:
    # 1 : #number of same app used per device:
    frame =pd.DataFrame( test.loc[:,['device_id','app']].groupby(['device_id','app']).size() , columns=['nbr_same_app']).reset_index()
    gc.collect()
    test=pd.merge(test,frame , on= ['device_id','app'] , how ='left' )
    #rectify the 1 in the number of same app  for devices with no events
    test.loc[test.app ==-1.0,'nbr_same_app'] =-1

    # 2 : #number of times of the same label used per device:
    frame =pd.DataFrame( test.loc[:,['device_id','label']].groupby(['device_id','label']).size() , columns=['nbr_same_label']).reset_index()
    gc.collect()
    test=pd.merge(test,frame , on= ['device_id','label'] , how ='left' )
    #rectify the 1 in the number of same app  for devices with no events
    test.loc[test.app ==-1.0,'nbr_same_label'] =-1

    # 3 : #number of times of the same category used per device:
    frame =pd.DataFrame( test.loc[:,['device_id','category']].groupby(['device_id','category']).size() , columns=['nbr_same_category']).reset_index()
    gc.collect()
    test=pd.merge(test,frame , on= ['device_id','category'] , how ='left')
    #rectify the 1 in the number of same app  for devices with no events
    test.loc[test.app ==-1.0,'nbr_same_category'] =-1
    
    
    #4 : number of occurence of each device 
    dict_device = dict(test.groupby(['device_id'])['app'].agg(np.size))
    test['device_occur'] = test.device_id.apply(dict_device.get)


     # 5 sum of labels
    dict_device = dict(test.loc[:,['device_id','nbr_same_label']].groupby(['device_id'])['nbr_same_label'].agg(np.sum))
    test['sum_of_labels'] = test.device_id.apply(dict_device.get)

    #6 sum of app
    dict_device = dict(test.loc[:,['device_id','nbr_same_app']].groupby(['device_id'])['nbr_same_app'].agg(np.sum))
    test['sum_of_app'] = test.device_id.apply(dict_device.get)
    
    #7 sum of category
    dict_device = dict(test.loc[:,['device_id','nbr_same_category']].groupby(['device_id'])['nbr_same_category'].agg(np.sum))
    test['sum_of_category'] = test.device_id.apply(dict_device.get)



    
    test.drop_duplicates('device_id' , keep ='first' , inplace =True )
    test.reset_index(drop=True , inplace=True)

    test.loc[test.app!=-1,['app','label','category']]=1
    test.drop(['device_id','label-occurence','nbr_same_app','nbr_same_label','nbr_same_category'],axis =1 , inplace = True)
    
 
    


    return train,test,target

train , test , y_train = load_data()

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
    "max_depth": 6,
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
                    100000,
                    evals= watchlist ,early_stopping_rounds=60)

    oof_test[:] += clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_iteration)
    oof_train[test_index]=clf.predict(xgb.DMatrix( X_val), ntree_limit=clf.best_iteration)
    
    
oof_test /= NFOLDS



xgb_predictions_test = pd.DataFrame(oof_test)
xgb_prediction_train = pd.DataFrame(oof_train)

xgb_predictions_test.to_csv('xgb_predictions_test.csv',index=None)
xgb_prediction_train.to_csv('xgb_prediction_train.csv',index=None)


print('-------- next stup : Stacking -----------')