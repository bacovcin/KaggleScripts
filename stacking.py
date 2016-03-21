import time
import pandas as pd
import numpy as np
from skxgboost import skxgboost
# from sklearn import svm
from sklearn import ensemble as ens
from sklearn import linear_model as lm
from sklearn import calibration
from sklearn.preprocessing import StandardScaler
# from sklearn import neighbors as nb
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression

# Define global variables

holdout_testing = True
holdout_prop = 0.1

# Types of models to fit
base_models = [
          (skxgboost({
                 "silent": 1,
                 "objective": "binary:logistic",
                 "booster": "gbtree",
                 "eval_metric": "logloss",
                 "eta": 0.01,
                 "min_child_weight": 1,
                 "subsample": 0.96,
                 "colsample_bytree": 0.45,
                 "max_depth": 10
                    }, 500), 20),
          (ens.ExtraTreesClassifier(n_estimators=700,
                                    max_features=50,
                                    criterion='entropy',
                                    min_samples_split=5,
                                    max_depth=50,
                                    min_samples_leaf=5), 20),
          ("PLS", 30)]

# Tuples of models and number of iterations to run on the output of the
# base models
ensemble_models = [
          (ens.RandomForestClassifier(n_estimators=128, max_depth=None,
                                      min_samples_split=1, n_jobs=-1), 50),
          (calibration.CalibratedClassifierCV(GaussianNB(),
                                              cv=10,
                                              method='isotonic'), 30),
          (calibration.CalibratedClassifierCV(lm.Perceptron(n_jobs=-1),
                                              cv=10,
                                              method='isotonic'), 30),
          (calibration.CalibratedClassifierCV(lm.LogisticRegression(n_jobs=-1),
                                              cv=2,
                                              method='isotonic'), 30)]

base_prop = 0.2                  # Proportion of data to use in base bootstraps

# Parameters for the beta function to determine ensemble bootstrap proportions
row_alpha = 2
row_beta = 10

col_alpha = 2
col_beta = 2

# Weighted average uses 1/(OOB_logloss ** weight_factor)
weight_factor = 2

# Output = (unweighted average * (1 - weight_scale) +
#          (weighted average * weight_scale)
weight_scale = .9

np.random.seed(69)


def fit_ensemble_model(classifiers, y,
                       trainx, testx):
    """ Fits the ensemble models to the output of the base models,
    and calculates the weighted and unweighted average of their results,
    and then combines them using the weighting score from the global variables
    """
    weights = []
    testProbs = []
    for classifier in classifiers:
        clf = classifier[0]
        number_of_ensembles = classifier[1]
        for i in range(number_of_ensembles):
            print(i)
            print(clf)
            boot_rows = np.random.choice(
                trainx.shape[0],
                round(trainx.shape[0] *
                      max([0.01,
                           min([np.random.beta(row_alpha,
                                               row_beta),
                                0.6])])),
                True)
            oob_rows = np.array(list(set(range(trainx.shape[0])) -
                                     set(boot_rows)))
            boot_cols = np.random.choice(
                trainx.shape[1],
                max([1,
                     round(trainx.shape[1] *
                           max([0.01,
                                min([np.random.beta(col_alpha,
                                                    col_beta), 0.99])]))]))
            tx = trainx[boot_rows[:, None],
                        boot_cols]          # Get the boot training scores
            print(tx.shape)
            ty = np.array(y[boot_rows])     # Get the boot training classes

            # Fit the classifier on the subset
            print("\t Fitting model...")
            trainMod = clf.fit(tx, ty)

            # Calculate the predicted probability of success for
            # the full training and test sets
            print("\t Calculating training probabilities...")
            TrainProbs = trainMod.predict_proba(trainx[:, boot_cols])
            ll = log_loss(y[oob_rows], TrainProbs[oob_rows])
            weights.append(1 / (ll ** weight_factor))
            TrainProbs = TrainProbs[:, 1]
            print("\t Calculating test probabilities...")
            testProbs.append(trainMod.predict_proba(testx[:,
                                                    boot_cols])[:, 1])
            print("\t Log_loss: " + str(ll))

    # Take the results of the ensembling models and calculate the weighted
    # and unweighted averages
    testArray = np.array(testProbs).transpose()

    weighted_mean = ((np.multiply(testArray,
                                  np.array(weights)).sum(axis=1)) /
                     np.array(weights).sum())

    unweighted_mean = testArray.mean(axis=1)

    # Combine weighted and unweighted mean in globally defined proportions
    outProb = ((unweighted_mean * (1 - weight_scale)) +
               (weighted_mean * weight_scale))

    return outProb


def fit_base_model(classifiers, fully, dummyY, trainx, testx):
    """ Takes a list of classifiers and/or PLS regression and
    does dimension reduction by returning the predictions of the classifiers
    or first two scores of the PLS regression on bootstrapped subsamples of
    the data."""

    trainProbs = []
    testProbs = []

    iterations = 0
    for clf in classifiers:
        for i in range(clf[1]):
            iterations += 1
            print(iterations)
            print(clf[0])
            train_rows = np.random.choice(trainx.shape[0],
                                          round(trainx.shape[0] * base_prop),
                                          True)
            oob_rows = list(set(range(trainx.shape[0])) - set(train_rows))
            print(len(train_rows))
            print(len(oob_rows))
            x = trainx[train_rows, :]
            if clf[0] == 'PLS':
                y = dummyY[train_rows, :]
                mod = PLSRegression().fit(x, y)
                trainscores = mod.transform(trainx)
                testscores = mod.transform(testx)
                trainProbs.append(trainscores[:, 0])
                trainProbs.append(trainscores[:, 1])
                testProbs.append(testscores[:, 0])
                testProbs.append(testscores[:, 1])
            else:
                y = fully[train_rows]
                print('\t Fitting model...')
                mod = clf[0].fit(x, y)
                print('\t Predicting training results...')
                tpreds = mod.predict_proba(trainx)
                trainProbs.append(list(tpreds[:, 1]))
                print('\t Predicting test results...')
                testProbs.append(list(mod.predict_proba(testx)[:, 1]))
                print('\t OOB score: ' + str(log_loss(fully[oob_rows],
                                                      tpreds[oob_rows, :])))
    return trainProbs, testProbs

if __name__ == "__main__":
    print('Loading Dataset...')
    train = pd.read_csv('../trainfe.csv')
    yload = pd.read_csv('../yfe.csv')
    y = np.array(yload)[:, 0]

    # Extract targets and reset to -1/1 coding
    y[y == 0] = -1
    dummyY = np.array(pd.get_dummies(yload.iloc[:, 0]))

    # Extract features
    xunscaled = np.array(train)

    # Load testing or create holdout set
    if holdout_testing:
        # Set a new seed for each run, so that we get different holdout sets
        # for each run, in order to get a sense of the distribution of
        # holdout scores.
        np.random.seed(round(time.time()))
        holdout_rows = np.random.choice(xunscaled.shape[0],
                                        round(holdout_prop *
                                              xunscaled.shape[0]))
        train_rows = list(set(range(xunscaled.shape[0])) -
                          set(holdout_rows))
        testxunscaled = xunscaled[holdout_rows, :]
        xunscaled = xunscaled[train_rows, :]
        holdout_y = y[holdout_rows]
        dummyY = dummyY[train_rows]
        y = y[train_rows]
        print(testxunscaled.shape)
        print(holdout_y.shape)
        print(xunscaled.shape)
        print(dummyY.shape)
        print(y.shape)
    else:
        test = pd.read_csv('../testfe.csv')
        testxunscaled = np.array(test)

    # Normalize features for non-treebased methods
    print('Normalizing...')
    ppMod = StandardScaler().fit(xunscaled)
    x = ppMod.transform(xunscaled)
    testx = ppMod.transform(testxunscaled)

    # Load test IDs
    testIDs = np.array(pd.read_csv('../testids.csv'))[:, 0]

    # Fit the initial bootstraps to data set
    print('Running Initial Bootstraps...')
    trainboot, testboot = fit_base_model(base_models,
                                         y, dummyY, x, testx)

    # Extract the unscaled outputs
    ustrainarray = np.array(trainboot).transpose()
    ustestarray = np.array(testboot).transpose()

    # Normalize the outputs of the base models

    ppMod2 = StandardScaler().fit(ustrainarray)
    trainarray = ppMod2.transform(ustrainarray)
    testarray = ppMod2.transform(ustestarray)

    print(trainarray)
    print(testarray)

    # Fit the ensemble models to the intiital bootstraps
    testprob = fit_ensemble_model(ensemble_models, y,
                                  trainarray, testarray)

    if holdout_testing:
        sprob = list(testprob)
        fprob = list(1 - np.array(testprob))
        print(np.array(sprob))
        print(np.array(fprob))
        print(holdout_y)
        print('Holdout Logloss: ' + str(log_loss(holdout_y,
                                                 np.array([fprob,
                                                          sprob]).transpose())))
    else:
        testoutput = [list(testIDs), list(testprob)]

        # Write the results to file
        print('Writting to File...')
        outfile = open('../stacking.csv', 'w')
        outfile.write('ID,PredictedProb\n')

        for i in range(len(testoutput[0])):
            outfile.write(str(testoutput[0][i]) + ',' +
                          str(min([1,
                                   max([testoutput[1][i],
                                        0])])) + '\n')
