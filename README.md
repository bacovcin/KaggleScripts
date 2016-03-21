A set of R/python tools for Kaggle competitions:

fe.R does basic feature engineering:
	-Adds a column equal to the number of NAs in a row
	-Adds a binary is/is not NA column for each variable
	-Implements median imputation for all variables
	-For categorical variables:
		-Generates binary dummy variables, if the number of 
		 levels is lower than a cut-off established at the 
		 beginning of the file and all levels exist in both 
		 the training and test set
		-Otherwise, replaces all factor levels with training
		 set proportion
	-Combines continuous and new categorical variables
	-Removes variables that can be predicted with a linear combination
	 of other variables (R2 is lower than a cut-off established at
	 top of file)
	-Saves the results as csv files in the directory below KaggleScripts

rw-bagging.py
	Randomly walks between different models with bootstrapping, using
	better scoring models more frequently and then averaging some of the
	final models (in accordance with parameters set at the top of the
	file)

stacking.py
	Implements a two tiered stacking system, that allows the results of
	either bootstrapped scikit learn models or PLS scores to be used as
	input to another round of bootstrap models, whose results are averaged
	together

skxgboost.py
	A wrapper around the python xgboost package that gives it fit and
	predict_proba methods like scikit learn models so that it can be
	used in the same code as scikit models
