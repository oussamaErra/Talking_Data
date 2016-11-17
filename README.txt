-------Errabia Oussama---------

my model is a 2-layer learning architecture model ( stacked model)

in the first level i train 2 models on the data with no events , and 4 models on the data with events

for the models on the data with no events:
		1- Xgboost
		2- Keras Nnet
for the models on the data with events :
		1-Xgboost
		2-keras Nnet
		3-Extratrees
		4-Random Forest
All models in 1st layers are trained using a 5 fold cross-validation technique using always the same fold indices.

the 2nd level i trained an Xgboost (possibly adding a keras nnet then average them which surly will outperform the current one )
in a 5 fold cross validation , It provided us the ability to calculate the score localy.

there was some features engineering but not that much ,possibility of adding even more features but 
they need to be tested so to see are they really helping the performance or the time of convergenece or it's all about adding noise to our data set.


to run my script:
first start by trainning the first layer models by runing:

 -keras_model.py ( with no events)
 -keras_with_events.py (with events)
 -the_xgboost_with_events.py (with events)
 -xgb_no_events.py (with no events)
 -Et_Rf_model.py ( the Exrtatrees and Random forest with events)

then run the stacking script:
  
-stacking model.py

for the first layer models : you need to specify the placements of the data to be used
for the second layer model you need to specify the placements of resulting data sets from the first layer models 

I hope my model will take enough intention from you.

My Best Regards.

