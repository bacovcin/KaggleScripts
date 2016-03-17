import time
import pandas as pd
import numpy as np
from skxgboost import skxgboost
# from sklearn import svm
from sklearn import ensemble as ens
from sklearn import linear_model as lm
from sklearn import calibration
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # , FastICA
# from sklearn import neighbors as nb
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB

# Define global variables
# Types of models to fit
models = [
          skxgboost({
                 "silent": 1,
                 "objective": "binary:logistic",
                 "booster": "gbtree",
                 "eval_metric": "logloss",
                 "eta": 0.01,
                 "min_child_weight": 1,
                 "subsample": 0.99,
                 "colsample_bytree": 0.65,
                 "max_depth": 11
                    }, 500),
          skxgboost({
                 "silent": 1,
                 "objective": "binary:logistic",
                 "booster": "gbtree",
                 "eval_metric": "logloss",
                 "eta": 0.01,
                 "min_child_weight": 1,
                 "subsample": 0.99,
                 "colsample_bytree": 0.45,
                 "max_depth": 11
                    }, 500),
          calibration.CalibratedClassifierCV(GaussianNB(),
                                             cv=10,
                                             method='isotonic'),
          calibration.CalibratedClassifierCV(lm.Perceptron(n_jobs=-1),
                                             cv=10,
                                             method='isotonic'),
          calibration.CalibratedClassifierCV(lm.LogisticRegression(n_jobs=-1),
                                             cv=2,
                                             method='isotonic'),
          ens.RandomForestClassifier(n_estimators=128, max_depth=None,
                                     min_samples_split=1, n_jobs=-1)]

init_models = 10     # Number of chances for initial model probabilities
feature_c = .008     # Rate at which each subsequent probability for initial

min_iters = 100       # Minimum mumber of iterations
stop_len = 50        # Number of worse models for stopping criterion
store_len = 60       # Number of models to store for ensemble

boot_alpha = 2       # Alpha for beta distribution on boot percentage
boot_beta = 5        # Beta for beta distribution on boot percentage

main_update = 0.1    # Main probability adjustment for features
minor_update = 0.05  # Probability adjustement for slightly worse models

eql_cutoff = 0.01    # Cut off for considering items equal
minor_cutoff = 0.02  # Cut off for minor reward
main_cutoff = 0.04   # Cut off for major penalty

model_major = 4      # Update for models that perform better
model_minor = 2      # Update for models that perform only slighly worse

np.random.seed(69)


def fit_ensemble_model(classifier, y,
                       trainx, testx, oobs,
                       col_nums, num_rows):
    """ Fits individual models for an ensemble. It takes a sklearn classifier,
    the full set of training classes, the full set of training PLS scores,
    the full set of test PLS score, the number of PLS scores to include,
    and the rows to train on (for bagging)"""
    tx = trainx[np.array(num_rows)[:, None],
                np.array(col_nums)]  # Get the subset training scores
    ty = np.array(y[num_rows])          # Get the subset training classes

    print("     Fitting model...")
    trainMod = classifier.fit(tx, ty)  # Fit the classifier on the subset

    # Calculate the predicted probability of success for
    # the full training and test sets
    print("     Calculating training probabilities...")
    TrainProbs = trainMod.predict_proba(trainx[:, col_nums])
    ll = log_loss(y[oobs], TrainProbs[oobs])
    TrainProbs = TrainProbs[:, 1]
    print("     Calculating test probabilities...")
    TestProbs = trainMod.predict_proba(testx[:,
                                             np.array(col_nums)])[:, 1]
    return (TrainProbs, TestProbs, ll)


def fit_ensemble(classifiers,
                 y, trainx, testx):
    """ Takes a list of classifiers, and features and
    runs a series of bootstraps which alter the probability
    of different models and features based on the oob scores,
    and then ensembles the final STORE_LEN models to produce
    a final estimate."""

    # Initialize best score by just taking the global average of the two
    # classes
    best_score = log_loss(y,
                          np.array([[len(y[y == -1])/len(y)] *
                                    len(y),
                                    [len(y[y == -1])/len(y)] *
                                    len(y)]).transpose())

    # Create the probability matrices for the random walk
    model_prob = np.array([init_models] * len(classifiers))
    feature_prob = [.99]
    cur_val = .99
    for i in range(1, trainx.shape[1]):
        if cur_val - 0.01 >= feature_c:
            cur_val -= feature_c
        feature_prob.append(cur_val)

    feature_prob = np.array(feature_prob)
    print(feature_prob)

    # Create variables to track the random walk bootstraping
    num_losers = 0
    TrainProbs = []
    TestProbs = []
    weights = []
    iteration = 0
    # Run bootstraps until the score does not improve after STOP_LEN runs
    while (num_losers <= stop_len) or (iteration <= min_iters):
        iteration += 1

        # Select a model type weighted by model probabilities
        clf_choice = np.random.choice(len(classifiers), 1,
                                      p=[x/model_prob.sum() for x in model_prob]
                                      )

        # Select features weighted by feature probabilities
        col_nums = []
        while len(col_nums) < 1:
            col_nums = np.arange(trainx.shape[1])[np.random.sample(
                trainx.shape[1]) <= feature_prob]

        # Select the bootstrap rows with a proportion
        # chosen by a beta function (min 0.01, max 0.9)
        # Min and max chosen to ensure both adequate train
        # and oob test data
        train_rows = np.random.choice(trainx.shape[0],
                                      round(trainx.shape[0] *
                                            min([0.9,
                                                 max([0.01,
                                                      np.random.beta(boot_alpha,
                                                                     boot_beta)
                                                      ])])),
                                      True)

        # Identify the out of bag rows for score calculation
        oob_rows = list(set(np.arange(trainx.shape[0])) - set(train_rows))

        # Fit the selected model
        # Print out current progress
        print('Iteration: ' + str(iteration) +
              '\tBest Score: ' + str(best_score) +
              '\tNum_losers: ' + str(num_losers))
        print(len(train_rows))
        print(len(col_nums))
        print(classifiers[clf_choice])
        results = fit_ensemble_model(classifiers[clf_choice], y,
                                     trainx, testx, oob_rows,
                                     col_nums, train_rows)

        print('Old Score: ' + str(best_score) +
              '\tCurrent Score: ' + str(results[2]) +
              '\tScore Diff: ' + str(results[2] - best_score))
        time.sleep(3)
        # Determine if the current model was better or worse than best
        if results[2] - best_score < eql_cutoff:
            # New best model: update best score and
            # improve probabilities of selected elements
            if results[2] < best_score:
                best_score = results[2]
                num_losers = 0
            model_prob[clf_choice] += model_major
            feature_prob[col_nums] = feature_prob[col_nums] + main_update
            if results[2] >= best_score:
                num_losers += 1
        else:
            # Model performed worse: increase count for stop criterion and
            # decrease probability of associated variables
            # If the model is close to best, only give a minor penalty
            num_losers += 1
            if results[2] - best_score < minor_cutoff:
                feature_prob[col_nums] = feature_prob[col_nums] + minor_update
                model_prob[clf_choice] += model_minor
            elif results[2] - best_score < main_cutoff:
                feature_prob[col_nums] = feature_prob[col_nums] + minor_update
                model_prob[clf_choice] -= model_minor
            else:
                model_prob[clf_choice] -= model_major
                feature_prob[col_nums] = feature_prob[col_nums] - main_update
        if model_prob[clf_choice] < 1:
            model_prob[clf_choice] = 1
        feature_prob = np.array(
            [max([.01, min([x, .99])]) for x in feature_prob]
        )
        print(model_prob)
        print(feature_prob)

        # Store the results for ensembling
        TestProbs.append(results[1])
        TrainProbs.append(results[0])
        weights.append(1/(results[2]))

        # Eliminate oldest results to keep only the final set
        if len(TestProbs) > store_len:
            TrainProbs.pop()
            TestProbs.pop()
            weights.pop()

    print("Ensembling Feature Set...")
    TestArray = np.array(TestProbs).transpose()
    TrainArray = np.array(TrainProbs).transpose()

    # Calculate the weighted average probability
    outProb = TestArray.mean(axis=1)

    # Print the predicted score of the weighted average on the training set
    # Since this is not a holdout, score is likely to be overly optimistic
    print('waLL: ' +
          str(log_loss(y,
                       TrainArray.mean(axis=1))))

    return outProb

if __name__ == "__main__":
    print('Loading Dataset...')
    train = pd.read_csv('../trainfe.csv')
    test = pd.read_csv('../testfe.csv')
    y = np.array(pd.read_csv('../yfe.csv'))[:, 0]

    # Extract targets and reset to -1/1 coding
    y[y == 0] = -1

    # Extract features
    xunscaled = np.array(train)
    testxunscaled = np.array(test)

    # Normalize features for non-treebased methods
    print('Normalizing...')
    ppMod = StandardScaler().fit(xunscaled)
    xscaled = ppMod.transform(xunscaled)
    testxscaled = ppMod.transform(testxunscaled)

    # Run PCA and ICA analysis to meet Naive Bayes assumptions
    print('Calculating PCA...')
    pcaMod = PCA().fit(xscaled)

    x = pcaMod.transform(xscaled)
    testx = pcaMod.transform(testxscaled)

    # Print the cumulative sum of the variance explained for each score
    evr = pcaMod.explained_variance_ratio_
    cum_sum = 0
    for i in range(len(evr)):
        cum_sum += evr[i]
        print(str(i) + ": {0:.5f}".format(cum_sum))

    # Load test IDs
    testIDs = np.array(pd.read_csv('../testids.csv'))[:, 0]

    # Fit the ensemble to data set
    print('Running Bootstraps...')
    testprob = fit_ensemble(models,
                            y, x, testx)

    testoutput = [list(testIDs), list(testprob)]

    # Write teh results to file
    print('Writting to File...')
    outfile = open('../rw-bagging.csv', 'w')
    outfile.write('ID,PredictedProb\n')

    for i in range(len(testoutput[0])):
        outfile.write(str(testoutput[0][i])+','+str(testoutput[1][i])+'\n')
