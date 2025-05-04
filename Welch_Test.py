#############################################################################
##### Welch tests

#############################################################################

#### Imports

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import combinations
from statannot import add_stat_annotation


#############################################################################
##### Input - output paths

Entradas = '../data'
Salidas = '../data'

#############################################################################
##### WELCH

def Welch_test(Data, Dimension_cluster, dimension_analisis):
    if ((Data.columns[-1] == "AZP")|(Data.columns[-1] == "K_Means")):
        Data[Dimension_cluster] = Data[Dimension_cluster].replace([0, 1, 2, 3, 4],[1, 2, 3, 4, 5])
        DATA = Data[[ Dimension_cluster, dimension_analisis]]

        Sub_01 = DATA[DATA[Dimension_cluster] == 1][dimension_analisis].dropna()
        Sub_02 = DATA[DATA[Dimension_cluster] == 2][dimension_analisis].dropna()
        Sub_03 = DATA[DATA[Dimension_cluster] == 3][dimension_analisis].dropna()
        Sub_04 = DATA[DATA[Dimension_cluster] == 4][dimension_analisis].dropna()
        Sub_05 = DATA[DATA[Dimension_cluster] == 5][dimension_analisis].dropna()

        Sub_01_err = 1.96*(Sub_01.std())/(np.sqrt(Sub_01.shape[0]))
        Sub_02_err = 1.96*(Sub_02.std())/(np.sqrt(Sub_02.shape[0]))
        Sub_03_err = 1.96*(Sub_03.std())/(np.sqrt(Sub_03.shape[0]))
        Sub_04_err = 1.96*(Sub_04.std())/(np.sqrt(Sub_04.shape[0]))
        Sub_05_err = 1.96*(Sub_05.std())/(np.sqrt(Sub_05.shape[0]))


        New_Data = pd.DataFrame([['01', Sub_01.mean(), Sub_01_err],
                                 ['02', Sub_02.mean(), Sub_02_err],
                                 ['03', Sub_03.mean(), Sub_03_err],
                                 ['04', Sub_04.mean(), Sub_04_err],
                                 ['05', Sub_05.mean(), Sub_05_err]],
                                columns=['x', 'y', 'yerr'])
    else:
        DATA = Data[[ Dimension_cluster, dimension_analisis]]

        Sub_01 = DATA[DATA[Dimension_cluster] == 1][dimension_analisis].dropna()
        Sub_02 = DATA[DATA[Dimension_cluster] == 2][dimension_analisis].dropna()
        Sub_03 = DATA[DATA[Dimension_cluster] == 3][dimension_analisis].dropna()
        Sub_04 = DATA[DATA[Dimension_cluster] == 4][dimension_analisis].dropna()
        Sub_05 = DATA[DATA[Dimension_cluster] == 5][dimension_analisis].dropna()

        Sub_01_err = 1.96*(Sub_01.std())/(np.sqrt(Sub_01.shape[0]))
        Sub_02_err = 1.96*(Sub_02.std())/(np.sqrt(Sub_02.shape[0]))
        Sub_03_err = 1.96*(Sub_03.std())/(np.sqrt(Sub_03.shape[0]))
        Sub_04_err = 1.96*(Sub_04.std())/(np.sqrt(Sub_04.shape[0]))
        Sub_05_err = 1.96*(Sub_05.std())/(np.sqrt(Sub_05.shape[0]))

        New_Data = pd.DataFrame([['01', Sub_01.mean(), Sub_01_err],
                                 ['02', Sub_02.mean(), Sub_02_err],
                                 ['03', Sub_03.mean(), Sub_03_err],
                                 ['04', Sub_04.mean(), Sub_04_err],
                                 ['05', Sub_05.mean(), Sub_05_err]],
                                columns=['x', 'y', 'yerr'])

    return DATA, New_Data

#############################################################################
##### WELCH Plots

def MedErr_graph (Data, Title):

    def formatter(x, pos):
        return str(round(x / 1e6, 1)) + " MDP"

    color = ['#A65E1A','#E9C27D','#B2B2B2','#80CDC1','#018571']

    sns.set(font_scale = 1.5, font="Algerian")
    sns.set_style("white")

    fig = plt.figure (figsize = (16,8), facecolor = "white")

    ax = sns.barplot(x = Data.x,
                     y = Data.y,
                     yerr = Data.yerr,
                     palette = color)
    ax.set(yticks=[Data.y.mean(),  Data.y.max()])
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.label.set_visible(False)
    plt.ylabel(' $\overline{X}_{Price}$',fontsize = 18)
    sns.despine()
    plt.savefig(Title, dpi=300, bbox_inches='tight')


#############################################################################
##### t-TEST & p-VALUE 

def testWelch(Data):

    dp_drup = dict(tuple(Data.drop_duplicates(inplace=False).groupby(Data.columns[0])))

    def t_test(pair):
        results= ttest_ind(dp_drup[pair[0]][Data.columns[-1]],
                           dp_drup[pair[1]][Data.columns[-1]])
        return (results)

    all_combinations = list(combinations(list(dp_drup.keys()), 2))
    t_test = pd.DataFrame([t_test(i) for i in all_combinations])
    Combina = pd.DataFrame(all_combinations, columns =['Ld', 'Jm'])

    New_Data = pd.concat([t_test, Combina], axis=1)

    New_Data = New_Data.replace({1:'01', 2:'02',
                                 3:'03', 4:'04',
                                 5:'05'})

    New_Data = New_Data[['Ld', 'Jm', 'statistic','pvalue']]

    return(New_Data)


#############################################################################
##### WELCH plots (p-VALUES)

def Welch_full (Table, Test, Title):

    def formatter(x, pos):
        return str(round(x / 1e6, 1)) + " MDP"

    color = ['#A65E1A','#E9C27D','#B2B2B2','#80CDC1','#018571']

    sns.set(font_scale = 1, font="Algerian")
    sns.set_style("white")

    fig = plt.figure (figsize = (16,8), facecolor = "white")

    ax = sns.barplot(x = Table.x,
                     y = Table.y,
                     yerr = Table.yerr,
                     palette = color)

    add_stat_annotation(ax,
                        x = Table.x,
                        y = Table.y,
                        order = Table.x,
                        box_pairs = Test[['Ld', 'Jm']].values.tolist(),
                        perform_stat_test = False,
                        pvalues = Test["pvalue"].values.tolist(),
                        test = None,
                        text_format = 'simple',
                        loc='outside',
                        verbose=2);

    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.label.set_visible(False)
    plt.ylabel(' $\overline{X}_{Price}$')
    plt.title(Title, fontsize = 12)
    sns.despine()
    plt.savefig(Title, dpi=300, bbox_inches='tight')


#############################################################################
##### Example

PRUEBA = pd.read_csv(Entradas + 'TSNE_DEPAS_DEPAS.csv')

DATA, TABLE = Welch_test(PRUEBA, 't_SNE', 'precio')

MedErr_graph(TABLE, 'TITULO SALIDA')

TEST = testWelch(DATA)

Welch_full (TABLE, TEST, 'TITULO SALIDA')
