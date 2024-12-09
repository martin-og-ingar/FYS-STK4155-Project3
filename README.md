# FYS-STK4155-Project3

## This repository contain the source code and report for project 3.

To run this project you can either:

- Run 'python evaluate.py' to evaluate the model on different hyper-parameters, and predict on the test data with the optimal hyper-parameters.
- Run 'python predict.py' to predict with the test data, using the pre-saved model with optimal hyper-parameters.

For both 'evaluate.py' and 'predict.py' you must add a argument, which is the model you want to use. This can be:

- 'ridge' for ridge regression,
- 'dt' for decision trees,
- 'rf' for random forest,
- or 'bagging' for bagging

Packages required to run this project can be found in req.txt, and can be installed with the command:
pip install -r req.txt

Project structure:

- The data folder includes datasets used and produced to and from this project.
- The figures folder includes figures produced from this project. The folder named after the models, includes graphs from each run. There are also the folders r2 and rmse, which includes heatmaps with the scores for each model. Random forest and bagging has a heatmap for each n_estimators value. Furthermore, the trees from decision tree are also plotted and saved in the trees folder. The test folder includes graphs for each model from when they were tested in the predict.py file (the final test).
- engineering_exploration folder includes files where engineering and exploration takes place, eg. preparing the dataset or testing other models.
- The evaluate.py facilitates the evaluation of each model, more specifically grid search over different hyper-parameter values on the models.
- The train.py trains a given model with time series cross validation.
- The predict.py file simply predicts on the test dataset for a given pretrained model (the optimal model from the evaluation).
- utils.py contains helper functions.
