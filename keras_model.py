

#keras nnet model without events


import numpy as np
np.random.seed(1991)
import datetime

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping


## Batch generators ##################################################################################################################################
def to_categorical(y, nb_classes=None):
    
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def multiclass_logloss(P, Y):
    npreds = [P[i][Y[i]-1] for i in range(len(Y))]
    score = -(1. / len(Y)) * np.sum(np.log(npreds))
    return score


def batch_generator(X, y, batch_size, shuffle):
    
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

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

    brand=pd.read_csv(path_brand_phne, dtype={'device_id': np.str} )
    brand.drop_duplicates('device_id', keep='first', inplace=True)
    brand['phone_brand'] = pd.factorize( brand['phone_brand'],sort=True)[0]
    brand['device_model'] = pd.factorize( brand['device_model'],sort=True)[0]   
    
    train_loader = pd.read_csv(path_train, dtype={'device_id': np.str})
    train = train_loader.drop(['age', 'gender'], axis=1)
    train['group'] = pd.factorize( train['group'],sort=True)[0] 
    train = pd.merge(train, brand, how='left', on='device_id', left_index=True )
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
train,test,y =load_data()




## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []
features = ['device_model','phone_brand']

for f in features:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)



del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(1200, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1000, init = 'he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(12, init = 'he_normal',activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    return(model)

## cv-folds
nfolds = 5
folds = StratifiedKFold(y, n_folds = nfolds, random_state = 0)

## train models
i = 0
nbags = 20
nepochs = 20
number_class=y.nunique()

pred_oob = np.zeros((xtrain.shape[0],number_class))
pred_test = np.zeros((xtest.shape[0],number_class))

#target to categorical
y_cat=to_categorical(y.values)

#start training 
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y_cat[inTr]
    xte = xtrain[inTe]
    yte = y_cat[inTe]
    pred = np.zeros((xte.shape[0],number_class))
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 34, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 1,validation_data=(xte.todense(),yte),callbacks=[EarlyStopping(patience=5)])
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])
    pred /= nbags
    pred_oob[inTe] = pred
    score = log_loss(yte, pred)
    i += 1
    print('Fold ', i, '- logloss:', score)
total_score=log_loss(y, pred_oob)
pred_test /= (nfolds*nbags)

print('Total - logloss:', total_score)


keras_predictions_test = pd.DataFrame(pred_test)
keras_prediction_train = pd.DataFrame(pred_oob)

keras_predictions_test.to_csv('keras_predictions_test.csv',index=None)
keras_prediction_train.to_csv('keras_prediction_train.csv',index=None)






