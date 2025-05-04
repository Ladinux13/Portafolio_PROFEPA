#############################################################################
##### OUTLIER DETECTION
##### LOCAL OUTLIER FACTOR (LOF)
#############################################################################

#### Imports

import numpy as np
import pandas as pd
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

### OUTLIER

from sklearn.neighbors import LocalOutlierFactor


#############################################################################
##### Input - output paths

Entradas = '../data'
Salidas = '../data'

#############################################################################
#####

INMUEBLES = gpd.read_file(Entradas + 'INMUEBLES_VENTA.shp')
HEX_CDMX = gpd.read_file(Entradas + 'HEX_CDMX.shp')


#############################################################################
#####  House outliers

CASAS_VENTA = INMUEBLES[INMUEBLES['Oferta'].isin(['Casa'])]

CLF_CASA = LocalOutlierFactor( n_neighbors = 300,
                               contamination ='auto')

X_CASAS = CASAS_VENTA[['precio','superficie','habitacion','baños']].values

Y_CASAS = CLF_CASA.fit_predict(X_CASAS)

DF_CASAS = pd.DataFrame(Y_CASAS, columns=['outlier'])

OUTLIER_CASAS = pd.DataFrame(CASAS_VENTA[[ 'id','precio',
                                    'superficie','habitacion',
                                    'baños']])

OUTLIER_CASAS['Outlier'] = DF_CASAS.values

OUTLIER_CASAS = OUTLIER_CASAS.query('Outlier == 1')

OUT_LIER = CASAS_VENTA.merge( OUTLIER_CASAS[['id','Outlier']],
                              left_on = 'id',
                              right_on = 'id',
                              how ='right')

OUT_LIER = OUT_LIER.to_crs(6372)

##### Hex codes

HEX_CASA = gpd.sjoin( HEX_CDMX,
                      OUT_LIER,
                      how="right",
                      op='contains')

HEX_CASA = HEX_CASA[HEX_CASA['hex'].notna()]

HEX_CASA = HEX_CASA.drop([ 'index_left', 'Outlier',
                           'geometry'], axis=1)

#############################################################################
#####  Apartment Outliers

DEPAS_VENTA = INMUEBLES[INMUEBLES['Oferta'].isin(['Departamento'])]

CLF_DEPA = LocalOutlierFactor( n_neighbors = 300,
                               contamination ='auto')

X_DEPA = DEPAS_VENTA[['precio','superficie','habitacion','baños']].values

Y_DEPA = CLF_DEPA.fit_predict(X_DEPA)

DF_DEPA= pd.DataFrame(Y_DEPA, columns=['outlier'])

OUTLIER_DEPA = pd.DataFrame(DEPAS_VENTA[[ 'id','precio',
                                          'superficie','habitacion',
                                          'baños']])

OUTLIER_DEPA['Outlier'] = DF_DEPA.values

OUTLIER_DEPA = OUTLIER_DEPA.query('Outlier == 1')

OUT_LIED = DEPAS_VENTA.merge( OUTLIER_DEPA[['id','Outlier']],
                              left_on = 'id',
                              right_on = 'id',
                              how ='right')

OUT_LIED = OUT_LIED.to_crs(6372)

##### Assign hex code

HEX_DPTO = gpd.sjoin( HEX_CDMX,
                      OUT_LIED,
                      how="right",
                      op='contains')

HEX_DPTO = HEX_DPTO[HEX_DPTO['hex'].notna()]

HEX_DPTO = HEX_DPTO.drop([ 'index_left', 'Outlier',
                           'geometry'], axis=1)


#############################################################################
##### Write as csv

### CASAS
HEX_CASA.to_csv(Salidas + "CASAS_EN_VENTA.csv", index = False)

### DEPARTAMENTOS
HEX_DPTO.to_csv(Salidas + "DEPAS_EN_VENTA.csv", index = False)
