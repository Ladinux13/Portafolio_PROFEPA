### ================================= Modelos de clasificación por ML ================================= ###
### ================================= ================================= =============================== ###
### ============================================ Librerias ============================================ ###

import numpy as np
import scipy.stats
import pandas as pd
import xgboost as xgb
import multiprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score 

import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 

### ============================================ RED NEURONAL ============================================ ###

import time
import keras
import multiprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from keras.optimizers import Adam, RMSprop, SGD

### ============================================ Warnings ============================================ ###

import warnings
warnings.filterwarnings('ignore')


### ================================= ================================= ================================= ###
### ====================================== Entrenamiento y Pruebas ====================================== ###
### ================================= ================================= ================================= ###

def Train_Test (Tabla, Normaliza):
    
    Label = Tabla['CLASE']
    Covar = Tabla.drop(columns = 'CLASE')
    TranS = Normaliza()
    
    Covar_Tr = pd.DataFrame(TranS.fit_transform(Covar),
                            columns = Covar.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(Covar_Tr,
                                                        Label,
                                                        test_size = 0.20,
                                                        random_state = 42)
    
    print("\n Test", X_train.shape, "\n Train", X_test.shape)

    return (X_train, X_test, y_train, y_test)


### ================================= ================================= ================================= ###
### ======================================== Clasificador XGBoost ======================================= ###
### ================================= ================================= ================================= ###


def XGBoost_Class(X_train, y_train, X_test, y_test):
    
    parametrs = {"subsample": [0.25, 0.50, 0.75, 1],
                 "max_depth":[None, 1, 3, 5, 10, 20, 30],
                 "learning_rate":[0.001, 0.01, 0.1],
                 "booster":['gbtree']
                 }
    
    idx_validacion = np.random.choice(X_train.shape[0],
                                      size = int(X_train.shape[0]*0.1),
                                      replace = False)
    
    X_Val = X_train.iloc[idx_validacion,:].copy()
    y_Val = y_train.iloc[idx_validacion].copy()
    
    X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()

    eval_set =  [(X_Val, y_Val)]
    
    estimador = xgb.XGBClassifier(n_estimators = 1000,
                                  n_jobs = multiprocessing.cpu_count() - 1,
                                  eval_metric = 'rmse',
                                  early_stopping_rounds = 10)
    
    model = GridSearchCV(estimator = estimador,
                         param_grid = parametrs, 
                         cv = RepeatedKFold(n_splits = 5, n_repeats = 1, random_state = 42),
                         scoring = 'accuracy',
                         verbose = 0)
    
    GridFit = model.fit(X = X_train_grid,
                        y = y_train_grid,
                        eval_set = eval_set,
                        verbose = 0)
    
    Resultado = pd.DataFrame(GridFit.cv_results_)

    T_Resultado = Resultado.filter(regex = '(param*|mean_t|std_t)').drop(columns = 'params').sort_values('mean_test_score', ascending = False)\
                                .head(4)
    
    Best_estimador = GridFit.best_estimator_

    y_hat = Best_estimador.predict(X = X_test)

    RMSE = mean_squared_error(y_true = y_test, y_pred = y_hat, squared = False)

    print("----------------------------")
    print(f"RMSE test:{round(RMSE, 3)}")
    print("----------------------------")

    return(GridFit, T_Resultado, y_hat)


### ================================= ================================= ================================= ###
### ================================= Clasificador Regresión Logisitica ================================= ###
### ================================= ================================= ================================= ###

def LogRegression_Class(X_train, y_train, X_test, y_test):
    
    Param_LogReg = [{'penalty':['l2'],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'C': 10**np.linspace(-3,3,20),
                     }]
    
    idx_validacion = np.random.choice(X_train.shape[0],
                                      size = int(X_train.shape[0]*0.1),
                                      replace = False)
    
    X_Val = X_train.iloc[idx_validacion,:].copy()
    y_Val = y_train.iloc[idx_validacion].copy()
    
    X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    
    estimador_LogReg = LogisticRegression(max_iter = 1000)
    
    model_LogReg = GridSearchCV(estimator = estimador_LogReg,
                                param_grid = Param_LogReg,
                                cv = RepeatedKFold(n_splits = 5, n_repeats = 1, random_state = 42),
                                scoring = 'accuracy',
                                verbose = 0
                                )
    
    GridFit = model_LogReg.fit(X = X_train_grid,
                               y = y_train_grid)
    
    LogReg_Resultados = pd.DataFrame(model_LogReg.cv_results_)
    
    T_Resultado = LogReg_Resultados.filter(regex = '(param*|mean_t|std_t)').drop(columns = 'params').sort_values('mean_test_score',
                                                                                                                 ascending = False).head(4)
    Best_estimador = GridFit.best_estimator_

    y_hat = Best_estimador.predict(X = X_test)

    RMSE = mean_squared_error(y_true = y_test, y_pred = y_hat, squared = False)

    print("----------------------------")
    print(f"RMSE test:{round(RMSE, 3)}")
    print("----------------------------")

    return(GridFit, T_Resultado, y_hat)


### ================================= ================================= ================================= ###
### ==================================== Clasificador Random Forest ===================================== ###
### ================================= ================================= ================================= ###


def RandomFor_Class(X_train, y_train, X_test, y_test):
    
    Paramgd_RanFor = {'n_estimators': [64, 128, 256, 512],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [2, 4, 8, 16, 32, 64, 128],
                      'criterion': ['gini', 'entropy']
                      }
    
    idx_validacion = np.random.choice(X_train.shape[0],
                                      size = int(X_train.shape[0]*0.1),
                                      replace = False)
    
    X_Val = X_train.iloc[idx_validacion,:].copy()
    y_Val = y_train.iloc[idx_validacion].copy()
    
    X_Val = X_train.iloc[idx_validacion,:].copy()
    y_Val = y_train.iloc[idx_validacion].copy()
    
    X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    
    estimador_RF = RandomForestClassifier(random_state = 42,
                                          n_jobs = multiprocessing.cpu_count()-1)
    
    Model_RF = GridSearchCV(estimator = estimador_RF,
                            param_grid = Paramgd_RanFor,
                            cv = RepeatedKFold(n_splits = 3, n_repeats = 1, random_state = 42),
                            scoring = 'accuracy',
                            refit = True,
                            verbose = 0,
                            return_train_score = True)
    
    GridFit = Model_RF.fit(X = X_train_grid,
                           y = y_train_grid)
    
    RandomFo_Resultados = pd.DataFrame(Model_RF.cv_results_)
    
    T_Resultado = RandomFo_Resultados.filter(regex = '(param*|mean_t|std_t)').drop(columns = 'params').sort_values('mean_test_score',
                                                                                                                 ascending = False).head(4)
    Best_estimador = GridFit.best_estimator_

    y_hat = Best_estimador.predict(X = X_test)

    RMSE = mean_squared_error(y_true = y_test, y_pred = y_hat, squared = False)

    print("----------------------------")
    print(f"RMSE test:{round(RMSE, 3)}")
    print("----------------------------")

    return(GridFit, T_Resultado, y_hat)


### ================================= ================================= ================================= ###
### =========================== Matriz de confusión y Reporte Clasificación ============================= ###
### ================================= ================================= ================================= ###

def Confu_Matrix (y_test, y_hat):
    
    Accuracy = accuracy_score(y_true = y_test,
                              y_pred = y_hat,
                              normalize = True)
    
    print("")
    print("-------------------")
    print (f"Test accuracy: {np.round(100* Accuracy, 3)} %")
    print("")
    print("-------------------")
    print(classification_report(y_test, y_hat))
    
    accuracy_dt = np.round(accuracy_score(y_test, y_hat), 2)
    misclassification_dt = np.round(1-accuracy_dt, 2)
    precision_dt = np.round(precision_score(y_test, y_hat), 2)
    recall_dt = np.round(recall_score(y_test, y_hat), 2)
    f1_score_dt = np.round(f1_score(y_test, y_hat), 2)
    
    print('Score')
    print('Accuracy:              {0:.2f}'.format(accuracy_dt))
    print('Misclassification:     {0:.2f}'.format(misclassification_dt))
    print('Precision:             {0:.2f}'.format(precision_dt))
    print('Recall:                {0:.2f}'.format(recall_dt))
    print('F1-Score:              {0:.2f}\n'.format(f1_score_dt))
    
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_hat, normalize='true'), annot=True, ax=ax)
    ax.set_title('Classification Confusion Matrix')
    ax.set_ylabel('True Value')
    ax.set_xlabel('Predicted Value')





### ================================= ================================= ================================= ###
### ====================================== Modelo Generador de Datos ==================================== ###
### ================================= ================================= ================================= ###

class FixableDataFrame(pd.DataFrame):
    def __init__(self, *args, fixed={}, **kwargs):
        self.__dict__["var_dictionary"] = fixed
        super(FixableDataFrame, self).__init__(*args, **kwargs)
    def __setitem__(self, key, value):
        out = super(FixableDataFrame, self).__setitem__(key, value)
        if isinstance(key, str) and key in self.__dict__["var_dictionary"]:
            out = super(FixableDataFrame, self).__setitem__(key, self.__dict__["var_dictionary"][key])

### =================================
### Generador de datos
### =================================

def generator(n, fixed={}, seed=0):
    if seed is not None:
        np.random.seed(seed)
    X = FixableDataFrame(fixed=fixed)

### ================================= ================================= ================================= ###
### ============================================ DATOS GENERALES ========================================= ###
### ================================= ================================= ================================= ###


    # Variable "GENERO" con 0:Hombre & 1: Mujer. La ocurrencia es de 70% hombres y 30% mujeres
    X["GENERO"] = np.random.choice([0,1], size=(n,), p = [0.70, 0.30])

    # Variable "EDAD" distribuida de manera uniforme en un rango de 18 a 35 años
    X["EDAD"] =  np.random.uniform(18, 35, size=(n,)) + (np.random.normal(0, 2, size=(n,)))

    # Variable "CEDPRO" con 0: Sin Cédula & 1: Con Cédula
    X["CEDPRO"] = np.random.choice([0,1], size=(n,), p = [0.30, 0.70]) 

    # Variable "YEAEMP" años empleados distribuida de manera uniforme en un rango de 0 a 6 años
    X["YEAEMP"] = np.random.uniform(0, 6, size=(n,)) + (np.random.normal(0, 2, size=(n,)))

    # Variable "SANSFP" con 0:Sin Sanciones & 1:Con Sanciones
    X["SANSFP"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 
    
    # Variable "OTRSAN" con 0:Sin otro tipo de Sanciones & 1: Con otro tipo de Sanciones
    X["OTRSAN"] = np.random.choice([0,1], size=(n,), p = [0.90, 0.10]) 

    # Variable "FMPENAL" con 0:familiar SIN antecedente penal & 1: Familiar CON antecedentes
    X["FMPENAL"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 

    # Variable "DEMJUD" con 0:SIN demanda judicial & 1: CON demanda 
    X["DEMJUD"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20])

    # Variable "FMDELIN" con 0:familiar SIN vínculos en la delincuencia organizada & 1:familiar CON vínculos en la delincuencia organizada
    X["FMDELIN"] = np.random.choice([0,1], size=(n,), p = [0.95, 0.05]) 
    
    # Variable TOTEMP como consecuencia de CEDPRO, YEAEMP, SANSFP, OTRSAN y U[ERROR]
    X["TOTEMP"] = (X["CEDPRO"] )+ (X["YEAEMP"]) + (X["SANSFP"]) + (X["OTRSAN"]) + (np.random.normal(0, 2, size=(n,)))

    # Variable ESTCIV como consecuencia de GENERO, EDAD y U[ERROR]
    X["ESTCIV"] = (X["GENERO"]) + (X["EDAD"]) + (np.random.normal(0, 2, size=(n,)))

    # Variable NIVEST como consecuencia de ESTCIV, GENERO, EDAD, CEDPRO, TOTEMP y U[ERROR]
    X["NIVEST"] = (X["ESTCIV"]) + (X["GENERO"]) + (X["EDAD"]) + (X["CEDPRO"]) + (X["TOTEMP"]) + (np.random.normal(0, 2, size=(n,)))

    # Variable NIVLAB como consecuencia de NIVEST, SANSFP, OTRSAN, TOTEMP, y U[ERROR]
    X["NIVLAB"] = (X["NIVEST"]) + (X["SANSFP"]) + (X["OTRSAN"]) + (X["TOTEMP"]) + (np.random.normal(0, 2, size=(n,)))

    # Variable ATPENAL como consecuencia de FMPENAL, DEMJUD, FMDELIN y U[ERROR]
    
    X["ATPENAL"] = (X["FMPENAL"]) + (X["DEMJUD"]) + (X["FMDELIN"]) + (np.random.normal(0, 2, size=(n,)))



### ================================= ================================= ================================= ###
### ====================================== Bienes Muebles e inmuebles ====================================== ###
### ================================= ================================= ================================= ###
# 
#     # Variable "FADINMU"
    X["FADINMU"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 

    # Variable "TPOINMB" con FADINMU
    X["TPOINMB"] = X["FADINMU"] + (np.random.normal(0, 2, size=(n,))) + np.random.choice([0,1], size=(n,), p = [0.80, 0.20])

    # Variable "TITINMB" con TPOINMB & FADINMU
    X["TITINMB"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) +  X["TPOINMB"]  +  X["FADINMU"] + (np.random.normal(0, 2, size=(n,)))

    # Variable "NOINMB" con TPOINMB & TPOINMB
    X["NOINMB"] = np.random.uniform(0, 6, size=(n,)) + X["TPOINMB"] +  X["TITINMB"] + (np.random.normal(0, 2, size=(n,)))
    
    # Variable "FADVEH"
    X["FADVEH"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 

    # Variable "TITVEHI" con FADVEH
    X["TITVEHI"] = X["FADVEH"] + np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) + (np.random.normal(0, 2, size=(n,)))

    # Variable "NOVEHI" con TITVEHI
    X["NOVEHI"] = np.random.uniform(0, 6, size=(n,)) + X["TITVEHI"] + (np.random.normal(0, 2, size=(n,)))


### ================================= ================================= ================================= ###
### ====================================== CTAS BANCARIAS  ============================================== ###
### ================================= ================================= ================================= ###
# 


    # Variable "TITCTAS"
    X["TITCTAS"] = np.random.choice([0,1], size=(n,), p = [0.40, 0.60]) 

    # Variable "TPOCTAS"
    X["TPOCTAS"] = np.random.choice([0,1], size=(n,), p = [0.40, 0.60]) 

    # Variable "DINFRA"
    X["DINFRA"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 

    # Variable "DEUFAM"
    X["DEUFAM"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 

    # Variable "DEUFAM"
    X["DINPRES"] = np.random.choice([0,1], size=(n,), p = [0.60, 0.40]) 

    # Variable "NOCTAS"
    X["NOCTAS"] = X["TITCTAS"] + X["TPOCTAS"] + (np.random.normal(0, 2, size=(n,)))
    
    # Variable "FADINMU"
    X["FADINMU"] = np.random.choice([0,1], size=(n,), p = [0.80, 0.20]) 


    # Variable "DEPOSITO" con "NOCTAS", "DINFRA", "DEUFAM" y "DINPRES"
    X["DEPOSITO"] = np.random.uniform(15000, 44240, size=(n,))  + X["NOCTAS"] \
                     + X["DINFRA"] + X["DEUFAM"] + X["DINPRES"]\
                          + (np.random.normal(0, 2, size=(n,)))

    # Variable "RETIRO" con "NOCTAS" 
    X["RETIRO"] = np.random.uniform(15000, 45000, size=(n,))  + X["NOCTAS"]  + (np.random.normal(0, 2, size=(n,)))

    
    X["TIPO_CLASS"] = np.where((X["DEPOSITO"] > X["RETIRO"]),0,1)
    
### ================================= ================================= ================================= ###
### ====================================== CLASE  ======================================================= ###
### ================================= ================================= ================================= ###

    # Variable CLASE como consecuencia de todas las anteriores 
    X["CLASE"] = scipy.special.expit((1/X["NIVEST"])\
                                     + (1/X["NIVLAB"])\
                                        + (X["SANSFP"])\
                                            + (X["OTRSAN"])\
                                                + (1/X["ATPENAL"])\
                                                    +(X["FMPENAL"])\
                                                          +(X["DEMJUD"])\
                                                            +(X["FMDELIN"])\
                                                                + (X["NOINMB"]) + (X["TIPO_CLASS"])\
                                                                    + np.random.normal(0, 1, size=(n,)))
    
    X["CLASE"] = scipy.stats.bernoulli.rvs(X["CLASE"])

    return X

### ================================= ================================= ================================= ###
### ========================================= FUNCION GENERADORA ======================================== ###
### ================================= ================================= ================================= ###

def Generadora(n):
    X_full = generator(n)
    Data = X_full.drop(['DEPOSITO', 'RETIRO','TIPO_CLASS'], axis = 1)
    return Data


### ================================= ================================= ================================= ###
### ==================================== Clasificador Red Neuronal ====================================== ###
### ================================= ================================= ================================= ###


def ANN():
    
    Clasf = Sequential()
    
    Clasf.add(Dense(32, activation = 'relu', kernel_initializer = 'uniform', 
                    input_shape = (27,)))
    Clasf.add(Dropout(0.3))
    
    Clasf.add(Dense(16, activation = 'relu', 
                    kernel_initializer = 'uniform'))
    Clasf.add(Dropout(0.3))
    
    Clasf.add(Dense(8, activation = 'relu', 
                    kernel_initializer = 'uniform'))
    Clasf.add(Dropout(0.3))
    
    Clasf.add(Dense(1, activation = 'sigmoid', 
                    kernel_initializer = 'uniform'))

    
    Clasf.compile(loss = 'binary_crossentropy', 
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    return(Clasf)
    

def Neuronal(X_train, y_train, X_test, y_test):
    
    idx_validacion = np.random.choice(X_train.shape[0],
                                      size = int(X_train.shape[0]*0.1),
                                      replace = False)
    
    X_Val = X_train.iloc[idx_validacion,:].copy()
    y_Val = y_train.iloc[idx_validacion].copy()
    
    X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    

    For_trial = {'batch_size': [64,128,256],
                  'epochs': [80,100,140]}
    
    Clasifica_Model = KerasClassifier(ANN,
                                      verbose = 0)
    
    Neural_Model = GridSearchCV(estimator = Clasifica_Model,
                                param_grid = For_trial,
                                n_jobs = multiprocessing.cpu_count() - 1,
                                cv = RepeatedKFold(n_splits = 5, 
                                                   n_repeats = 1, 
                                                   random_state = 42),
                               verbose = 0)
    
    
    NwFit = Neural_Model.fit(X = X_train_grid,
                             y = y_train_grid,
                             validation_data = (X_Val, y_Val), 
                             verbose = 0)
    
    
    ANN_Pred = Neural_Model.best_estimator_
    
    y_hat = ANN_Pred.predict(X = X_test)
    
    RMSE = mean_squared_error(y_true = y_test, y_pred = y_hat, squared = False)

    print("----------------------------")
    print(f"RMSE test:{round(RMSE, 3)}")
    print("----------------------------")
    
    
    return(NwFit, y_hat)

### ================================= ================================= ================================= ###
### ================================= ================================= ================================= ###


### ================================= ================================= ================================= ###
### ================================== REGULARIZACIÓN DE CLASES ========================================= ###
### ================================= ================================= ================================= ###

def Optimiza_Class (X_train, y_train, X_test, y_test, pickle, ThOpt_metrics):

    y_proba_train = pickle.predict_proba(X_train)[:,1]
    threshold = np.round(np.arange(0.01, 0.85, 0.015), 2)

    if ThOpt_metrics == 'Kappa':
        tscores = []
        for thresh in threshold:
            scores = [1 if x >= thresh else 0 for x in y_proba_train]
            kappa = metrics.cohen_kappa_score(y_train, scores)
            tscores.append((np.round(kappa, 4), thresh))
        tscores.sort(reverse = True)
        thresh = tscores[0][-1]
    elif ThOpt_metrics == 'ROC':
        fpr, tpr, threshold_roc = metrics.roc_curve(y_train, y_proba_train, pos_label = 1)
        specificity = (1 - fpr)
        roc_dist = (( 2*tpr*specificity )/(tpr + specificity))
        thresh = threshold_roc[np.argmax(roc_dist)]

    y_proba_test= pickle.predict_proba(X_test)[:,1]

    scores = [ 1 if x >= thresh else 0 for x in y_proba_test ]
    auc = metrics.roc_auc_score(y_test, y_proba_test)
    kappa = metrics.cohen_kappa_score(y_test, scores)

    print ('Threshold: %.3f, Kappa: %.3f, AUC test-set: %.3f'% (thresh, kappa, auc))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, scores, normalize='true'), annot = True, ax = ax)
    ax.set_title('Classification Confusion Matrix')
    ax.set_ylabel('True Value')
    ax.set_xlabel('Predicted Value')


    Accuracy = accuracy_score(y_true = y_test,
                              y_pred = scores,
                              normalize = True)
    
    print("")
    print("-------------------")
    print (f"Test accuracy: {np.round(100* Accuracy, 3)} %")
    print("")
    print("-------------------")
    print(classification_report(y_test, scores))
    
    accuracy_dt = np.round(accuracy_score(y_test, scores), 2)
    misclassification_dt = np.round(1-accuracy_dt, 2)
    precision_dt = np.round(precision_score(y_test, scores), 2)
    recall_dt = np.round(recall_score(y_test, scores), 2)
    f1_score_dt = np.round(f1_score(y_test, scores), 2)
    
    print('Score')
    print('Accuracy:              {0:.2f}'.format(accuracy_dt))
    print('Misclassification:     {0:.2f}'.format(misclassification_dt))
    print('Precision:             {0:.2f}'.format(precision_dt))
    print('Recall:                {0:.2f}'.format(recall_dt))
    print('F1-Score:              {0:.2f}\n'.format(f1_score_dt))
    