#############################################################################
##### XGBOOST, SHAP, Confidence intervals
#############################################################################
#####  Imports

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#############################################################################
##### Figure size
# ==============================================================================
mpl.rcParams["figure.figsize"] = (10,6)
sns.set_context("paper")

#############################################################################
#####  Standarization
# ==============================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer


import multiprocessing
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor


#############################################################################
##### SHAP
shap.initjs()

#############################################################################
#####  DROP COLUMNS

def DROP_COLUMN (BASE_USO):
    feature_drop = ['hex', 'Entidad_cvgeo', 'Entidad_n', 'Municipio_cvgeo',
                    'Municipio_n', 'Colonia_clave', 'Colonia_n', 'Colonia_CP', 'Tipo',
                    'Oferta','Longitud','Latitud']
    datasets = [BASE_USO]
    for df in datasets:
        df.drop(feature_drop, axis = 1, inplace = True)
    return (BASE_USO)


#############################################################################
##### Transform input data
# ==============================================================================
def tran_values(Tabla_uso, Transformacion, testsize, rstate):
    ### Input:
    ### -> Base de datos
    ### -> Tipo de transformación de datos
    ### -> Tamaño de la muestra para la prueba (Test size)
    ### -> Aleatorización (Random state)
    Etiqueta = np.log1p(Tabla_uso['precio'])
    Var_explica = Tabla_uso.drop(['precio'],axis=1)
    SS = Transformacion()
    V_Scaler = pd.DataFrame(SS.fit_transform(Var_explica),columns = Var_explica.columns)
    X_train, X_test, y_train, y_test = train_test_split(V_Scaler, Etiqueta, test_size=testsize,
                                                        random_state = 42, shuffle = True)

    print ('Shapes', '\n', 'X_train: ', X_train.shape, '\n', 'y_train: ', y_train.shape, '\n', 'X_test: ', X_test.shape,
       '\n', 'y_test: ', y_test.shape,)
    ### Output:
    ### -> Muestra de variables explicativas para entrenamiento (X_train)
    ### -> Muestra de etiquetas para entrenamiento (y_train)
    ### -> Muestra de variables explicativas para prueba (X_test)
    ### -> Muestra de etiquetas para prueba (y_test)
    ### -> Se imprime la forma de los datos
    return (X_train, y_train, X_test, y_test)



#############################################################################
#####  XGBOOST
# ==============================================================================

def XGBtest (Xtrain ,ytrain, nstimators, nsplits, nrepeats):
    ### Input:
    ### -> Muestra de variables explicativas para entrenamiento (X_train)
    ### -> Muestra de etiquetas para entrenamiento (y_train)
    ### -> Número máximo de estimadores
    ### -> Número de pliegues (división de la muestra)
    ### -> Número de veces que es necesario repetir el validador cruzado.



    # part 01: Hyperparameters
    # ==============================================================================
    param_grid = {'max_depth'       : [None, 1, 3, 5, 10, 20],
                  'subsample'        : [0.5, 1],
                  'learning_rate'    : [0.001, 0.01, 0.1],
                  'booster'          : ['gbtree']
                 }
# ==================================================================================
    # part 02: validation data
    # ==============================================================================
    np.random.seed(123)
    idx_validacion = np.random.choice(Xtrain.shape[0], size= int(Xtrain.shape[0]*0.1), replace=False)
    X_val = Xtrain.iloc[idx_validacion, :].copy()
    y_val = ytrain.iloc[idx_validacion].copy()
    X_train_grid = Xtrain.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
    y_train_grid = ytrain.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()

# ==================================================================================
    # Part 03: training params
    # metodo = .fit()
    # ==============================================================================
    fit_params = {"early_stopping_rounds" : 10,
                  "eval_metric"           : "rmse",
                  "eval_set"              : [(X_val, y_val)],
                  "verbose"               : 0
                 }
# ==================================================================================
    # Part 04: grid search
    # ==============================================================================
    grid = GridSearchCV(estimator=XGBRegressor(n_estimators = nstimators, random_state = 123),
                        param_grid = param_grid,
                        scoring    = 'neg_root_mean_squared_error',
                        n_jobs     = multiprocessing.cpu_count() - 1,
                        cv         = RepeatedKFold(n_splits=nsplits, n_repeats= nrepeats , random_state=123),
                        refit      = True,
                        verbose    = 0,
                        return_train_score = True
                       )




    GridFit = grid.fit(X = X_train_grid, y = y_train_grid, **fit_params)

# ==================================================================================
    # Part 05: # results
    # ==============================================================================
    resultados = pd.DataFrame(GridFit.cv_results_)
    Tresultados = resultados.filter(regex = '(param.*|mean_t|std_t)')     .drop(columns = 'params')     .sort_values('mean_test_score', ascending = False)     .head(4)
    ### Output:
    ### ->  grid.fit en validación cruzada, mejores metricas de aprendizaje
    ### -> Tabla de resultados con los mejores parametros
    return (GridFit,Tresultados)


#############################################################################
##### Best parameters on trainig data

# ==============================================================================
def BestParam (Xtest, ytest, GridFit):
    ### Input:
    ### -> Varianles explicativas de prueba
    ### -> Variable predictiva de prueba
    ### -> Resultados en grid.fit, en el aprednizaje de los datos
# ==================================================================================
    # ==============================================================================
    print("----------------------------------------")
    print("Mejores hiperparámetros encontrados (cv)")
    print("----------------------------------------")
    print(GridFit.best_params_, ":", GridFit.best_score_, GridFit.scoring)

# ==================================================================================
    # ==============================================================================
    n_arboles_incluidos = len(GridFit.best_estimator_.get_booster().get_dump())
    print("----------------------------------------")
    print(f"Número de árboles incluidos en el modelo: {n_arboles_incluidos}")

# ==================================================================================
    # Estimate error on validation data
    # ==============================================================================
    F_Model = GridFit.best_estimator_
    predictor = F_Model.predict(data = Xtest)
    rmse = mean_squared_error( y_true = ytest, y_pred = predictor, squared = False )
    print("----------------------------------------")
    print(f"El error RMSE de test es: {round(rmse, 3)}")
    print("----------------------------------------")

    ### Output:
    ### -> Mejor estimador en grid.fit, mejores parametros de aprendizaje (F Model)
    ### -> Valores de predicción (Y_prediction)
    return (F_Model, predictor)


#############################################################################
#####  SHAP

def SHAP_INFO(F_Model,X_test):
    explainer = shap.TreeExplainer(F_Model)
    shap_values = explainer.shap_values(X_test)
    return (shap_values)

#############################################################################
#####  SHAP Swarm plots

def SHAP_ABEJAS(shap_values,X_test):
    shap.summary_plot(shap_values, X_test, show = False, cmap = 'copper_r')
    plt.savefig(dpi = 300, bbox_inches = 'tight')

### VIOLIN
def SHAP_VIOLIN(shap_values,X_test):
    shap.summary_plot(shap_values, X_test, plot_type='violin')



### Predictor importance from random forest
# ==============================================================================
def Ypred_Import(FModl, Xtrainer, Fdrop):
    ### Input:
    ### -> Mejores resultados en los parametros de aprendizaje ".best_estimator_"
    ### -> Variables explicativas de entrenamiento (X_train)
    ### -> Variables omitidas en set de entrenamiento (Drop  X_train)
    importancia_predictores = pd.DataFrame(
        {'predictor': Xtrainer.drop(Fdrop, axis = 1).columns,
         'importancia': FModl.feature_importances_})
    print("Importancia de los predictores en el modelo")
    print("-------------------------------------------")
    Ta_impo = importancia_predictores.sort_values('importancia', ascending=False)
    ### Output:
    ### -> Dataframe de importancia de variables explicativas en el modelo de predicción de manera ascendente.
    return (Ta_impo)


### Predictor importance under permutations
# ==============================================================================

def Permu_Import (FModl, Xtrain, ytrain, nrepeat, rstate):
    ### Input:
    ### -> Mejores resultados en los parametros de aprendizaje ".best_estimator_"
    ### -> Variables explicativas de entrenamiento (X_train)
    ### -> Variables omitidas en set de entrenamiento (X_train)
    ### -> Número de repeticiones
    ### -> Random state en permutaciones
    importancia = permutation_importance(
        estimator    = FModl,
        X            = Xtrain,
        y            = ytrain,
        n_repeats    = nrepeat,
        scoring      = 'neg_root_mean_squared_error',
        n_jobs       =  multiprocessing.cpu_count() - 1,
        random_state = rstate)
    df_importancia = pd.DataFrame({k: importancia[k] for k in ['importances_mean', 'importances_std']})
    df_importancia['feature'] = Xtrain.columns
    print("Importancia bajo permutaciones")
    print("-------------------------------------------")
    Per_Impo = df_importancia.sort_values('importances_mean', ascending=False)
    ### Output:
    ### -> Parametros de importancia de permutaciones
    ### -> Tabla relacionada a variables explicativas y  su importancia por permutaciones"
    return (importancia, Per_Impo)


### Permutations plot
# ==============================================================================
def Graph_Permu (Xtrain, Fdrop,importancia):
    ### Imput:
    ### -> Variables explicativas de entrenamiento (X_train)
    ### -> variables omitidas en el set de entrenamiento (F_drop)
    ### -> Importancia cde las variables calculada en las permutaciones
    fig, ax = plt.subplots()
    sorted_idx = importancia.importances_mean.argsort()
    ax.boxplot(importancia.importances[sorted_idx].T,
               vert   = False,
               labels = Xtrain.drop(Fdrop,axis=1).columns[sorted_idx] )
    ax.set_title('Importancia de los predictores (train)')
    ax.set_xlabel('Incremento del error tras la permutación')
    fig.tight_layout();
    ### Output:
    ### -> Grafica de importancia de predictores vs incremento de error tras permutación


### MAPE
# ==============================================================================
def Mape(y_true, y_pred):
    ### Input:
    ### -> Variable predictiva (etiqueta) de prueba (y_test)
    ### -> Resultados de predicción del modelo (y_hat)
    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            if actual[j] != 0:
                res[j] = (actual[j] - predicted[j]) / actual[j]
            else:
                res[j] = predicted[j] / np.mean(actual)
        return res
#=====================================================================================
    MAE =mean_absolute_error(np.asarray((y_true)), np.asarray((y_pred)))
    MAE_0 =mean_absolute_error(np.asarray(np.expm1(y_true)), np.asarray(np.expm1(y_pred)))
#=====================================================================================
    MSE = mean_squared_error(np.asarray((y_true)), np.asarray((y_pred)), squared=True)
    MSE_0 = mean_squared_error(np.asarray(np.expm1(y_true)), np.asarray(np.expm1(y_pred)), squared=True)
#=====================================================================================
    RMSE = np.sqrt(MSE)
    RMSE_0 = np.sqrt(MSE_0)
#=====================================================================================
    MAPE = np.mean(np.abs(percentage_error(np.asarray((y_true)), np.asarray((y_pred))))) * 100
    MAPE_0 = np.mean(np.abs(percentage_error(np.asarray(np.expm1(y_true)), np.asarray(np.expm1(y_pred))))) * 100
#=====================================================================================
    print("Performance measures")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print ('Mean absolute error (log1p) :', round(MAE, 3), )
    print("-------------------------------------------")
    print ('Mean Squared Error (log1p) :', round(MSE, 3),)
    print("-------------------------------------------")
    print ('Root Mean Squared Error (log1p) :', round(RMSE, 3),)
    print("-------------------------------------------")
    print ('Mean Absolute Percentage Error  (log1p) :', round(MAPE, 3), '%')
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print ('Mean absolute error (Normal) :', round(MAE_0, 3), )
    print("-------------------------------------------")
    print ('Mean Squared Error (Normal) :', round(MSE_0, 3),)
    print("-------------------------------------------")
    print ('Root Mean Squared Error(Normal) :', round(RMSE_0, 3),)
    print("-------------------------------------------")
    print ('Mean Absolute Percentage Error (Normal) :', round(MAPE_0, 3), '%')
    print("-------------------------------------------")
    ### Output:
    ### -> Metrica del error porcentual medio absoluto
    return (MAPE, MAPE_0)


### Prediction plots
# ==============================================================================
def Predicciones(ytest, yhat):
    predicciones = ytest.reset_index()
    predicciones = predicciones.rename(columns={"id": "Id_Inmueble", "precio": "Actual"})
    predicciones['Prediccion'] = pd.Series(np.reshape(yhat, (yhat.shape[0])))
    predicciones['diff'] = predicciones['Prediccion'] - predicciones['Actual']

# ==============================================================================
    plt.figure(figsize=(18,5))
    plt.plot(predicciones.Actual,'bo', markersize=4, color='#00A287', alpha=0.7, label='Actual')
    plt.plot(predicciones.Prediccion, linewidth=0.6, color='#FF6700', linestyle ="-", alpha=0.7, label='Prediccion')
    plt.legend()
    plt.title('Actual and prediction values')
# ==============================================================================
    plt.figure(figsize=(18,5))
    sns.distplot(predicciones['diff'],
                 kde=True,
                 rug=True,
                 rug_kws={"color": "#01939A", "alpha": 1, "linewidth": 0.2, "height":0.1},
                 kde_kws={"color": "#FF4540", "alpha": 0.3, "linewidth": 2, "shade": True})
    plt.title('Distribution of differences between actual and prediction')

# ==============================================================================
    plt.figure(figsize=(18,5))
    sns.regplot(x=predicciones.Actual, y=predicciones.Prediccion,
                line_kws={"color":"#9040D5","alpha":1,"lw":4},
                marker="+",
                scatter_kws={"color": "#A68F00"})
    plt.title('Relationship between actual and predicted values')
    plt.show()


    return (predicciones)


### Prediction intervals

# ==============================================================================
### LOSS & UPPER QUANTILE MODEL

def LoUp_Loss (Xtrain ,ytrain, v_alpha, nstima, nsplit, nrept):
    ### Input:
    ### -> Muestra de variables explicativas para entrenamiento (X_train)
    ### -> Muestra de etiquetas para entrenamiento (y_train)
    ### -> Número máximo de estimadores
    ### -> Número de pliegues (división de la muestra)
    ### -> Número de veces que es necesario repetir el validador cruzado.
    param_grid = {'max_features'  : ['auto', 'sqrt', 'log2'],
              'max_depth'     : [None, 1, 3, 5, 10, 20],
              'subsample'     : [0.5, 1],
              'learning_rate' : [0.001, 0.01, 0.1]
             }
    grid = GridSearchCV(
        estimator  = GradientBoostingRegressor(
                        loss                ="quantile",
                        alpha               = v_alpha,
                        n_estimators        = nstima,
                        random_state        = 123,
                        # Activación de la parada temprana
                        validation_fraction = 0.1,
                        n_iter_no_change    = 5,
                        tol                 = 0.0001
                    ),
        param_grid = param_grid,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits=nsplit, n_repeats=nrept, random_state=123),
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )
    LoUp_M = grid.fit(X = Xtrain, y = ytrain)

    # ==================================================================================
    # Part 05: # Results
    # ==============================================================================
    resultados = pd.DataFrame(LoUp_M.cv_results_)
    LoUp_Table = resultados.filter(regex = '(param.*|mean_t|std_t)') \
    .drop(columns = 'params') \
    .sort_values('mean_test_score', ascending = False) \
    .head(4)
    ### Output:
    ### ->  grid.fit en validación cruzada, mejores metricas de aprendizaje
    ### -> Tabla de resultados con los mejores parametros
    return (LoUp_M,LoUp_Table)


### 
### Quantile Loss results
# ==============================================================================
def Best_Qua (Xtest, ytest, GridFit):
    # Mejores hiperparámetros por validación cruzada
    print("----------------------------------------")
    print("Mejores hiperparámetros encontrados (cv)")
    print("----------------------------------------")
    print(GridFit.best_params_, ":", GridFit.best_score_, GridFit.scoring)
    # Número de árboles del modelo final (early stopping)
    # ==============================================================================
    print("----------------------------------------")
    print(f"Número de árboles del modelo: {GridFit.best_estimator_.n_estimators_}")
    print("----------------------------------------")
    # Error de test del modelo final
    # ==============================================================================
    F_LowUp = GridFit.best_estimator_
    LwUp_predi = F_LowUp.predict(X = Xtest)
    rmse = mean_squared_error(
        y_true  = ytest,
        y_pred  = LwUp_predi,
        squared = False
    )
    print("----------------------------------------")
    print(f"El error (rmse) de test es: {round(rmse, 3)}")
    print("----------------------------------------")
    return (F_LowUp, LwUp_predi)
### Output:
    ### -> Mejor estimador en grid.fit, mejores parametros de aprendizaje (F Model)
    ### -> Valores de predicción (Y_prediction)

# ==============================================================================
def Frame_Predic (T_Predic,yUpper,yLow):
    ### Input:
    ### -> Tabla de predicciones XGboost con campos index, actual, prediccion y diferencia (y_prediction)
    ### -> Predicciones de perdida en cuantil inferior (y_Low)
    ### -> Predicciones de perdida en cualtil superior (y_upper)
    T_Prediccions = T_Predic.copy()
    T_Prediccions['y_Upper'] = pd.Series(np.reshape(yUpper, (yUpper.shape[0])))
    T_Prediccions['y_Low'] = pd.Series(np.reshape(yLow, (yLow.shape[0])))
    T_export = T_Prediccions.copy()
    T_export['Actual'] = np.expm1(T_export.Actual)
    T_export['Prediccion'] = np.expm1(T_export.Prediccion)
    T_export['diff'] = T_export['Actual'] - T_export['Prediccion']
    T_export['y_Low'] = np.expm1(T_export.y_Low)
    T_export['y_Upper'] = np.expm1(T_export.y_Upper)
    T_export = T_export.rename(columns={"Prediccion": "Y_Pred",
                                        "diff": "Diff",
                                        "y_Low": "Y_Low",
                                       "y_Upper": "Y_Upper"})
    return (T_export, T_Prediccions)
# ==============================================================================


def Graph_Trace(T_Pred,HTML_Name):

    trace1=go.Scatter(x = T_Pred.index, y = np.expm1(T_Pred.y_Upper),
                      name = 'high',
                      fill = None,
                      mode = 'lines',
                      line=dict(color='#00BB3F', width=.5))

    trace2=go.Scatter(x = T_Pred.index, y = np.expm1(T_Pred.y_Low),
                      name = 'low',
                      fill = 'tonexty',
                      mode = 'lines',
                      line=dict(color='#00BB3F', width=.5))

    trace3 = go.Scatter(x=T_Pred.index, y = np.expm1(T_Pred.Actual),
                        mode='markers',
                        marker=dict(symbol='circle', color='#FF7F00', size = 3),
                        name = "Actual")

    trace4 = go.Scatter(x=T_Pred.index, y= np.expm1(T_Pred.Prediccion),
                        mode='markers',
                        marker=dict(symbol='circle', color='#044367', size = 3),
                        name="Prediction")

    data=[trace1,trace2,trace3,trace4 ]
    fig = go.Figure(data=data)

    fig.update_layout(hovermode='x unified')
    fig['layout']['xaxis']['title']='Inmueble'
    fig['layout']['yaxis']['title']='Precio'


    fig.update_layout( title="House Prices XGBOOST", legend_title="Simbología",
                      font=dict( family="Averta light, monospace",
                                size=10,
                                color="black") )

    fig.update_xaxes(showspikes=True, spikecolor="purple", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="purple", spikethickness = 1)
    fig.update_layout(spikedistance=1000, hoverdistance=100)

    fig.show()
    fig.write_html(HTML_Name)
