# ppi_prediction
## Requirements
Following Python packages are needed:

- pandas >= 0.23.4
- numpy >= 1.15.2
- sklearn >= 0.20.0
- keras >= 2.2.2
- tensorflow >= 1.10.0

## Implementation
3 approaches have been implemented for solving this task. 

The first one is a Support Vector Regression. In order to optimize its parameters a grid search has been used. The other models are simple Neural Networks with 1 and 2 hidden layers. 

A A Jupyter notebook made in Colab is attached with the 3 experiments and results for these approaches.
Also, a script called *main.py* is provided. It has 2 modalities: train and predict. With the first one, we train a desired model and save it into models folder. With predict mode, 5 entries of the dataset are predicted using a stored model.

This command is used to train a model:
````
python main.py --model model_option --train_predict train
````
(For model_option, it is possible to choose between: svr, nn_1 and nn_2)

In order to predict (for this case, 5 random samples are selected from the provided dataset), this command is used:
````
python main.py --model model_option --train_predict predict
````

## Future improvements
In order to reduce the mean squared error, several options may be followed.

- Use transfer learning with Neural Networks. Due to lack of data, is very difficult to fit a good model. An idea is to pick a model developed for other task but related and trained with a lot of data, like house pricing prediction, and then use that model (or several trained layers of them) to transform original small dataset features into others and then train few layers with our target dataset. This pretrained model works as an embedding layer that transforms original feature space into a new one that may fit better with few data.

- Related with transfer learning, instead of picking a pretrained model, a large dataset can be used to train an embedding model in a first stage, which has the same purpose like the pretrained model in the point above. Once the embedding model is trained, it is used to transform the original feature space into the new one. 