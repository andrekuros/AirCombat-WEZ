#!/usr/bin/env python
# coding: utf-8

# # WEZ Model Generation
# This notebook implements the pipeline for Weapon Engagement Zone (WEZ) model generation from the paper:
# 
# Optimized Prediction of Weapon Effectiveness in BVR Air Combat Scenarios Using Enhanced Regression Models
# 
# 
# ## Objectives
# - Develop and evaluate WEZ models using data from experiments.
# - Compare the performance of multiple regression techniques.
# 
# ---

#%%%
# ---
# ## Experiment Configuration
# 
# This section defines all the parameters required for the experiment. 
# 
# ### Key Sections:
# 1. **Experiment Tag and Range Type**:   
#    - `rangeType`: Specifies the output range to be modeled. Options include:
#      - `RMax`: Maximum range.
#      - `RNez`: No Escape Zone range.
#      - `Wez`: Multi-Objective Model for both Ranges
# 
# 2. **Data Splitting Parameters**:
#    - `testSplitMode`: Determines how the dataset is divided into training and testing sets. Options:
#      - `FixedTestSize`: Fixed number of training and test samples.
#      - `RandomSplit`: Randomly splits data based on a specified ratio.
#      - `AlternateTest`: Uses an alternate dataset for testing.
#    - `trainingSize` and `testSize`: Define the sizes of training and test datasets (applicable for `FixedTestSize`).
#    - `dataSplitRatio`: Ratio for splitting data (used for `RandomSplit` mode).
# 
# 3. **Normalization**:
#    - If `normalize` is `True`, the features will be normalized (scaled) to improve model performance.
# 
# 4. **Model Configuration**:
#    - Includes options for:
#      - Polynomial interaction degrees (`interactionsDegrees`).
#      - Feature reduction factors (`reductionsFactors`).
#      - Number of folds for cross-validation (`folds`).
#      - List of regressors to evaluate (`regressors`).
#          - "Lasso", "Ridge", "ElasticNet", "LinReg",\
#            "MLP32", "MLP128", "MLP256",\
#             "SVR_RBF", "SVR_Poly", "SVR_Linear"\
#             "RF"
# 
# 5. **Data Preprocessing Options**:
#    - Configures preprocessing:
#      - (R)  Adding relative red heading features (`relativeRedHdg`).
#      - (Da) Augmenting the dataset (`argumentation`).
#      - (T)  Applying sine and cosine transformations to angular features (`sinCos`).
# 
# 6. **Test Parameters**:
#    - Reserved for custom testing or advanced configurations.
# 
# ---
# 
# ### How to Use:
# 
# 1. Update the configuration values in the block below to customize your experiment settings.
# 2. For list-based configuration parameters, you can specify multiple values to explore different configurations in a single run.
# 3. After finalizing the configuration, proceed with the model training and evaluation steps.
# 4. Partial results will be displayed in the notebook, while the complete results will be saved in the `Output` folder.
# 

#Experiment identification
runTag = "Test01"

# Range Type
rangeType = "RMax"  # Specifies the type of output range to model (e.g., "RMax", "RNez", "Wez").

# Data Splitting Configuration
testSplitMode = "FixedTestSize" # Mode of splitting the data:
                                # - "FixedTestSize": Fixed training and test sizes.
                                # - "RandomSplit": Random split with a specified ratio.
                                # - "AlternateTest": Use alternate dataset for testing.

trainingSize = 700               # Number of samples in the training dataset (applicable for FixedTestSize).
testSize = 200                   # Number of samples in the testing dataset (applicable for FixedTestSize).
dataSplitRatio = 0.20            # Ratio for splitting data (applicable for RandomSplit).

# Normalization Flag
normalize = True  # Whether to normalize all features (MinMaxScaler or StandardScaler can be used later).

# Model Configuration
models_config = { 
    'interactionsDegrees': [3,4,5],             # Polynomial interaction degrees to explore.
    'reductionsFactors': [100],             # Number of features to retain after reduction.
    'folds': 5,                             # Number of folds for cross-validation.
    'regressors': ["Lasso", "Ridge"],       # List of regression models to evaluate. Examples:                                 
}

# Data Preprocessing Options
data_preprocess = {        
    'R' :  [True],   # Whether to include relative heading 
    'T' :  [True],   # Whether to include sine and cosine transformations of angular data
    'Da':  [True],   # Whether to augment the dataset
} 

# Test Parameters
test_params = [-1]  # Additional parameters to pass for testing specific configurations

if rangeType == "RMax":    
    dataFile = "RandomExperiment_1000_RMAX.csv"     
    multi_obj = False
    out_vars = ['RMax']
elif rangeType == "RNez":
    dataFile = "RandomExperiment_1000_NEZ.csv"     
    multi_obj = False
    out_vars = ['RNez']
elif rangeType == "Wez":
    dataFile = "RandomExperiment_1000_WEZ.csv"     
    multi_obj = True
    out_vars = ['RNez','RMax']



# ## Load the required Libraries and Functions 

# Required Libraries
import pandas as pd
import numpy as np
import math
import time
import itertools
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet 

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error  

import warnings
warnings.filterwarnings("ignore")


def prepareDataFromASA(fileName, R = True,\
                        Da = True, size = -1,  \
                        T = True, multiObj = False, rangeType = "WEZ"):
        
    fulldataDf = pd.read_csv(fileName)   
    
    if size > 0:
        fulldataDf = fulldataDf.sample(size)
        
    dataDf = fulldataDf[['BL_Speed', 'RD_Speed', 'rad', 'RD_Hdg', 'BL_Alt', 'RD_Alt']]
    
    if not multiObj:
        dataDf[rangeType] = fulldataDf[rangeType].replace(-1,0)        
        dataDf = dataDf[dataDf[rangeType] >= 2] 
    else:
        dataDf['RNez'] = fulldataDf['RNez'].replace(-1,0)                
        dataDf['RMax'] = fulldataDf['RMax'].replace(-1,0)                
        dataDf = dataDf[dataDf['RNez'] >= 2]         
        dataDf = dataDf[dataDf['RMax'] >= 2] 

    
    if Da:                
        
        dataDf_Arg = dataDf.copy()
        dataDf_Arg['rad'] = dataDf_Arg.apply(lambda x: -1 * x.rad , axis=1)  
        dataDf_Arg['RD_Hdg'] = dataDf_Arg.apply(lambda x: -1 * x.RD_Hdg , axis=1)         
        dataDf = pd.concat([dataDf, dataDf_Arg], axis=0, ignore_index=True)        
            
    dataDf['diffAlt'] = dataDf.apply(lambda x: x.BL_Alt - x.RD_Alt, axis=1)      
                
    if R:        
        dataDf['relRedHdg'] = dataDf.apply(lambda x: (x.RD_Hdg-x.rad) , axis=1) 
        if T:
            dataDf['cosRel'] = dataDf.apply(lambda x: math.cos(x.relRedHdg*math.pi/180), axis=1)
            dataDf['sinRel'] = dataDf.apply(lambda x: math.sin(x.relRedHdg*math.pi/180), axis=1) 
            dataDf = dataDf.drop(['relRedHdg'], axis=1)              
    
    else:     
        dataDf['relRedHdg'] = dataDf.apply(lambda x: 0-x.RD_Hdg , axis=1)
        if T:
            dataDf['cosRel'] = dataDf.apply(lambda x: math.cos(x.relRedHdg*math.pi/180), axis=1)
            dataDf['sinRel'] = dataDf.apply(lambda x: math.sin(x.relRedHdg*math.pi/180), axis=1)
            dataDf['cosRel2'] = dataDf.apply(lambda x: math.cos(x.rad*math.pi/180), axis=1)
            dataDf['sinRel2'] = dataDf.apply(lambda x: math.sin(x.rad*math.pi/180), axis=1)
            dataDf = dataDf.drop(['relRedHdg'], axis=1)  
            #dataDf = dataDf.drop(['rad'], axis =1)                                                          
    
    dataDf = dataDf.drop(['RD_Hdg'], axis =1)     
    dataDf = dataDf.drop(['RD_Alt'], axis =1) 
        
    return dataDf

def split_train_test(X, Y, mode, fold, trainingSize, testSize, dataSplitRatio, DataSetTest=None):
    """
    Splits the data into training and testing sets based on the mode.

    Parameters:
        X (pd.DataFrame): Features.
        Y (pd.DataFrame): Target.
        mode (str): Split mode (RandomSplit, AlternateTest, FixedTestSize).
        fold (int): Current fold index.
        trainingSize (int): Size of training data.
        testSize (int): Size of testing data.
        dataSplitRatio (float): Ratio for RandomSplit.
        DataSetTest (pd.DataFrame, optional): Alternate test dataset.

    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    if mode == "RandomSplit":
        return train_test_split(X, Y, test_size=dataSplitRatio, random_state=fold)
    elif mode == "AlternateTest":
        X_train = X[:trainingSize]
        Y_train = Y[:trainingSize]
        shuffled_test = shuffle(DataSetTest, random_state=fold)
        X_test = shuffled_test.drop(["maxRange"], axis=1)[:600]
        Y_test = shuffled_test.maxRange[:600]
        return X_train, X_test, Y_train, Y_test
    elif mode == "FixedTestSize":
        return train_test_split(X, Y, test_size=testSize, train_size=trainingSize, random_state=fold)
    else:
        raise ValueError(f"Unsupported split mode: {mode}")

def get_regressor(name, fold, multi_obj=False):
    """
    Returns the specified regression model with appropriate parameters.
    """
    if name == "MLP32":
        return MLPRegressor(random_state=fold, max_iter=500000, activation='tanh', hidden_layer_sizes=(32, 32))
    elif name == "MLP128":
        return MLPRegressor(random_state=fold, max_iter=500000, activation='tanh', hidden_layer_sizes=(128, 128))
    elif name == "MLP256":
        return MLPRegressor(random_state=fold, max_iter=500000, activation='tanh', hidden_layer_sizes=(256, 256))
    elif name == "Ridge":
        return Ridge(alpha=0.02, max_iter=500000)
    elif name == "Lasso":
        return Lasso(alpha=0.00025, max_iter=500000)
    elif name == "ElasticNet":
        return ElasticNet(alpha=0.00025, l1_ratio=0.5,  max_iter=500000)
    elif name == "RF":
        return RandomForestRegressor(n_estimators=50, max_depth=20, random_state=fold, n_jobs=10)
    elif name == "LinReg":
        return LinearRegression(n_jobs=15, fit_intercept=False)
    elif name == "SVR_Poly":
        svr = SVR(kernel="poly", C=5000, gamma="auto", degree=1, epsilon=0.1, max_iter=500000)
        return MultiOutputRegressor(svr) if multi_obj else svr
    elif name == "SVR_RBF":
        svr = SVR(kernel="rbf", C=5000, gamma="auto", epsilon=0.1, max_iter=500000)
        return MultiOutputRegressor(svr) if multi_obj else svr
    elif name == "SVR_Linear":
        svr = SVR(kernel="linear", C=5000, gamma="auto", epsilon=0.1, max_iter=500000)
        return MultiOutputRegressor(svr) if multi_obj else svr
    else:
        raise ValueError(f"Unknown Model: {name}")

def perform_feature_reduction(X_trainS, X_testS, model, polyReductionFactors, regressor, powers, Y_train):
    """
    Reduces features based on model coefficients.

    Parameters:
        X_trainS (pd.DataFrame): Training dataset with features.
        X_testS (pd.DataFrame): Test dataset with features.
        model: Trained model with accessible coefficients.
        polyReductionFactors (int): Number of features to retain.
        regressor (str): Type of regressor (e.g., "Ridge", "Lasso").
        powers (np.ndarray): Powers of polynomial features.
        Y_train (pd.DataFrame): Target variable for training.

    Returns:
        tuple: (X_train_reduced, X_test_reduced, effects, updated_powers)
    """
    effects = []

    if polyReductionFactors > 0:
        # Extract coefficients based on the model type
        if regressor == "Ridge":
            coefs = model.coef_[0]
        elif regressor == "Lasso":
            coefs = model.coef_
        else:
            print(f"Error: Model {regressor} does not accept parameters reduction")
            exit(1)
        
        # Pair features with coefficients and sort by importance
        coefsAux = [[X_trainS.columns[i], c] for i, c in enumerate(list(coefs))]
        coefsAux = sorted(coefsAux, key=lambda x: -abs(x[1]))

        if coefsAux:
            # Identify insignificant features to drop
            insig = [x[0] for x in coefsAux[polyReductionFactors:]]
            totalCoefs = sum([abs(x[1]) for x in coefsAux])

            # Record effects for the remaining features
            dictEf = {"total": totalCoefs}
            for x in coefsAux:
                dictEf[x[0]] = x[1] / totalCoefs
            effects.append(dictEf)

            # Drop insignificant features
            X_trainS = X_trainS.drop(insig, axis=1)
            X_testS = X_testS.drop(insig, axis=1)

            # Update powers
            mask = np.ones(powers.shape[0], dtype=bool)
            mask[insig] = False
            powers = powers[mask]

            # Refit the model on reduced features
            model.fit(X_trainS, Y_train)

            # Recompute coefficients and effects after reduction
            if regressor == "Ridge":
                coefs = model.coef_[0]
            elif regressor == "Lasso":
                coefs = model.coef_

            coefsAux = [[X_trainS.columns[i], c] for i, c in enumerate(list(coefs))]
            coefsAux = sorted(coefsAux, key=lambda x: -abs(x[1]))
            totalCoefs = sum([abs(x[1]) for x in coefsAux])

            dictEf = {"total": totalCoefs}
            for x in coefsAux:
                dictEf[x[0]] = x[1] / totalCoefs
            effects.append(dictEf)
        else:
            print("Fail to Reduce Params")
            exit(1)

    return X_trainS, X_testS, effects, powers

def print_configuration_summary(rangeType, dataFile, testSplitMode, trainingSize, testSize, 
                                 dataSplitRatio, multi_obj, out_vars, models_config, data_preprocess, normalize):
    """
    Prints a detailed summary of the experiment configuration.
    
    Parameters:        
        rangeType (str): Range type for the experiment.
        dataFile (str): Dataset file name.
        testSplitMode (str): Mode of splitting the dataset.
        trainingSize (int): Number of training samples.
        testSize (int): Number of test samples.
        dataSplitRatio (float): Ratio of data splitting for random split.
        multi_obj (bool): Indicates if the task is multi-objective.
        out_vars (list): Output variables for prediction.
        models_config (dict): Model configurations.
        data_preprocess (dict): Data preprocessing configurations.
        normalize (bool): Whether all features should be normalized.        
    """
    print("\n" + "="*50)
    print("                Experiment Configuration Summary")
    print("="*50)    
    print(f"Range Type                 : {rangeType}")
    print(f"Dataset File               : {dataFile}")
    print(f"Test Split Mode            : {testSplitMode}")
    
    if testSplitMode == "RandomSplit":
        print(f"Data Split Ratio           : {dataSplitRatio:.2f}")
    else:
        print(f"Training Set Size          : {trainingSize}")
        print(f"Test Set Size              : {testSize}")
        
    print(f"Multi-Objective            : {multi_obj}")
    print(f"Output Variables           : {', '.join(out_vars)}")
    print("\nModel Configurations:")
    print(f"  Interaction Degrees      : {models_config['interactionsDegrees']}")
    print(f"  Reduction Factors        : {models_config['reductionsFactors']}")
    print(f"  Number of Folds          : {models_config['folds']}")
    print(f"  Selected Regressors      : {', '.join(models_config['regressors'])}")
    print("\nPreprocessing Configurations:")
    for key, values in data_preprocess.items():
        print(f"  {key:5}: {values}")
    print(f"\nNormalize Features     : {normalize}")    
    print("="*50 + "\n")

def bootstrap_ci(data, metric, n_bootstrap=5000, max_iterations=1000, min_variation=0.00001):
    bootstrap_samples = []
    prev_ci_lower, prev_ci_upper = None, None
    
    for i in range(max_iterations):
        bootstrap_sample = np.random.choice(data[metric], size=len(data), replace=True)
        bootstrap_samples.append(np.mean(bootstrap_sample))
        
        ci_lower, ci_upper = np.percentile(bootstrap_samples, [2.5, 97.5])
        
        if prev_ci_lower is not None and prev_ci_upper is not None:
            variation = max(abs(ci_lower - prev_ci_lower), abs(ci_upper - prev_ci_upper))
            if variation < min_variation:
                break
        
        prev_ci_lower, prev_ci_upper = ci_lower, ci_upper
    
    return (ci_lower, ci_upper)

def evalData(regressor, output_vars, ref_vel, X, Y,  timeReps = 100):
    
    #Eval on Training set
    start = time.time()
    
    for i in range(timeReps):
        y_pred = regressor.predict(X)     
    predTime = time.time() - start  
    
    eval_results = {}   
    
    for i, output_name in enumerate(output_vars):
        
        Y_item  = np.array(Y[output_name])                
        
        if len(output_vars) > 1:
            y_pred_item = np.array(y_pred[:,i])
        else:
            y_pred_item = np.array(y_pred).flatten()
                    
        
        predDiff = Y_item - y_pred_item        
        maxError = max( abs(predDiff)) 
        
        if isinstance(ref_vel, np.ndarray):                    
            time_err = np.mean(abs((predDiff) / ref_vel))
        else:
            time_err = np.mean(abs((predDiff) / 550))
            
        rel_err =  np.mean(abs((predDiff) / Y_item))               
        mae = mean_absolute_error(Y_item, y_pred_item)
        rmse = math.sqrt(mean_squared_error(Y_item, y_pred_item)) 
        
        eval_results.update({output_name + "_MAE" : mae, 
                             output_name + "_RMSE" : rmse,
                             output_name + "_time_err": time_err,
                             output_name + "_relative_err" : rel_err,                             
                             output_name + "_max_error" :maxError})
    
    eval_results['MAE']  = mean_absolute_error(Y, y_pred)    
    eval_results['RMSE'] = math.sqrt(mean_squared_error(Y, y_pred)) 
    eval_results['MAPE'] =  np.mean(abs((predDiff) / Y_item))   
        
    eval_results.update({"pred_time" : predTime/len(Y) * 1000 , "eval_reps" : timeReps} )

    return (y_pred, eval_results)


# Print summary of configurations
print_configuration_summary(    
    rangeType=rangeType,
    dataFile=dataFile,
    testSplitMode=testSplitMode,
    trainingSize=trainingSize,
    testSize=testSize,
    dataSplitRatio=dataSplitRatio,
    multi_obj=multi_obj,
    out_vars=out_vars,
    models_config=models_config,
    data_preprocess=data_preprocess,
    normalize=normalize,    
)


# ---
# ## Run the Experiments
#Run the experiments according to the specified configurations
metricsSummary = []

proccess_combinations = list(itertools.product(*data_preprocess.values()))

for params in proccess_combinations:
    
    config = dict(zip(data_preprocess.keys(), params))        
    dataSet = prepareDataFromASA('./Data/' + dataFile, multiObj=multi_obj, rangeType = rangeType, **config)        
    
    for regressor in models_config['regressors']:       
        for interactionsDegree in models_config['interactionsDegrees']:
            for polyReductionFactors in models_config['reductionsFactors']:                
                for it, test_param in enumerate(test_params):
                                    
                    metricsTest = []                    
                    
                    for fold in range(models_config['folds']):
                        
                        if not multi_obj:
                            X = dataSet.drop([rangeType], axis=1)
                            Y = dataSet[[rangeType]]
                        else:
                            X = dataSet.drop(["RNez", "RMax"], axis=1)
                            Y = dataSet[["RNez", "RMax"]]                                                                            
                        
                        
                        X_train, X_test, Y_train, Y_test = split_train_test(
                            X, Y, testSplitMode, fold, trainingSize, testSize, dataSplitRatio
                        )                       
                        
                        X_trainS = X_train                    
                        X_testS = X_test                    
                                                                                    
                        # Create interaction terms (interaction of each regressor pair + polynomial)                        
                        if interactionsDegree > 1:
                            if regressor == "LinReg" or regressor == "Lasso" or regressor == "Ridge" or regressor == "ElasticNet":                            
                                interaction = PolynomialFeatures(degree=interactionsDegree, include_bias=False, interaction_only=False)                                                  
                                X_trainS = pd.DataFrame(interaction.fit_transform(X_trainS), columns=interaction.get_feature_names_out(input_features=list(map(str,list(X_trainS.columns)))))   
                                X_testS = pd.DataFrame(interaction.fit_transform(X_testS), columns=interaction.get_feature_names_out(input_features=list(map(str,list(X_testS.columns)))))
                                powers = interaction.powers_
                                cols = X_trainS.columns                                                           

                        if normalize:
                            #scaler = StandardScaler()  
                            scaler = MinMaxScaler()
                            scaler.fit_transform(X_trainS)                            
                            X_trainS = scaler.transform(X_trainS)    
                            X_testS = scaler.transform(X_testS)                                            
                            X_trainS = pd.DataFrame(X_trainS)
                            X_testS = pd.DataFrame(X_testS)
                        else:
                            X_trainS = pd.DataFrame(X_trainS.values)
                            X_testS  = pd.DataFrame(X_testS.values)
                                                                                                                            
                        model = get_regressor(regressor, fold, multi_obj)                        
                                
                        pipeline = Pipeline([('model', model)])                        
                        start = time.time()                                                            
                        pipeline.fit(X_trainS,Y_train)                    
                        fitTime = time.time() - start
                        
                        print(
                            f'\r--- ({fold+1}/{models_config["folds"]}) {rangeType} | '
                            f'{regressor} | '
                            f'IntDeg({interactionsDegree}) | '                            
                            f'{("TestParam(" + str(test_param) + ") | ") if test_param != -1 else ""}'
                            f'preProc('
                            f'{"R" if config["R"] else ""}'
                            f'{"T" if config["T"] else ""}'
                            f'{"Da" if config["Da"] else ""})'
                            f'{(" | Reduct(" + str(polyReductionFactors) + ")") if polyReductionFactors > 0 else ""}'
                            f' ---', 
                            end=""
                        )

                                                
                        
                        if polyReductionFactors > 0:
                            X_trainS, X_testS, effects, powers = perform_feature_reduction(
                                X_trainS, X_testS, model, polyReductionFactors, regressor, powers, Y_train
                            )
                                                
                        y_pred, eval_data = evalData(pipeline, out_vars, np.array(pd.DataFrame(X_test.BL_Speed)), X_testS, Y_test )
                                                    
                        test_data = {"type"                 : "Test_Data", 
                                    "regressor"             : regressor,
                                    "preprocess_params"     : params,
                                    "interaction_degree"    : interactionsDegree, 
                                    "reduction_factors"     : polyReductionFactors,
                                    "test_param"            : test_param,
                                    "training_size"         : trainingSize
                                    }
                        test_data.update(eval_data)
                        test_data["fit_time"] = fitTime
                                                                    
                        metricsTest.append(test_data)
                                                                
                    metricsDf = pd.DataFrame(metricsTest)
                    
                    summary = {}                    
                    for col in metricsDf.columns:                                                                      
                        if np.issubdtype(metricsDf[col].dtype, np.number):                            
                            summary[col + '_mean'] = metricsDf[col].mean()
                            summary[col + '_std']  = metricsDf[col].std()
                            
                            if col.find('MAE') != -1 or col.find('RMSE') != -1:                                
                                bt_ci = bootstrap_ci(metricsDf, col)
                                summary[col + '_bs_ci_low']  = bt_ci[0]
                                summary[col + '_bs_ci_up']  = bt_ci[1]
                        else:                            
                            summary[col] = metricsDf[col][0]
                                                            
                    print(f'\nMAE: {summary["MAE_mean"]:.3f} CI({summary["MAE_bs_ci_low"]:.3f}, {summary["MAE_bs_ci_up"]:.3f})' )                                       
                    print(f'MAPE: {summary["MAPE_mean"]:.3f}')                  
                    print(f'Fit/Pred: {summary["fit_time_mean"]:.3f} [s]' , end="")
                    print(f' / {summary["pred_time_mean"]:.3f} [ms]\n' )
                    
                                                            
                    metricsSummary.append(summary)
        
                                                        
summaryMetricDf = pd.DataFrame(metricsSummary)
regressor_names = "_"
for reg in models_config['regressors']:
    regressor_names += reg + "_"
    
output_file = ".\Output\Summary_" + runTag + "_" + rangeType + regressor_names + str(models_config['folds'])  + ".csv"
summaryMetricDf.to_csv( output_file ,index=False, header=True, mode='w')

print( f'Summary Results saved to {output_file}' )




# %%
