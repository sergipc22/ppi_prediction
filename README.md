# ppi_prediction
3 approaches have been implemented for solving this task. The first one is a Support Vector Regression. In order to optimize its parameters a grid search has been used. The other models are simple Neural Networks with 1 and 2 hidden layers. 
A notebook is attached with the 3 experiments and results for these approaches.
Also, a script called *main.py* is provided. It has 2 modalities: train and predict. With the first one, we train a desired model and save it into models folder. With predict mode, 5 entries of the dataset are predicted using a stored model.

This command is used to train a model:
````
python main.py --model svr --train_predict train
````
(For model option, it is possible to choose between: svr, nn_1 and nn_2)

In order to predict, this command is used:
````
python main.py --model svr --train_predict predict
````