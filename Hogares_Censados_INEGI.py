#############################################################################
##### HOGARES CENSALES INEGI
#############################################################################

#### LIBRERIAS

from h3 import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

import warnings
warnings.filterwarnings('ignore')

def id2poly(hex_id):
    boundary = [x[::-1] for x in h3.h3_to_geo_boundary(hex_id)]
    return Polygon(boundary)

#############################################################################
##### ARCHIVOS: ENTRADAS - SALIDAS

Entradas = ''
Salidas = ''

#############################################################################
#####

CDMX = gpd.read_file(Entradas + 'CDMX.shp')

MANZA_CDMX = gpd.read_file(Entradas + 'INV2020_IND_PVEU_MZA_09.shp')

HOGAR = pd.read_excel(Entradas +'RESAGEBURB_09XLSX20.xlsx')

#############################################################################
##### CREACIÓN DE HEXAGONOS CON BASE A UNA GEOMETRIA DE ÁREA

hexs = h3.polyfill( CDMX.geometry[0].__geo_interface__,
                    9, ### RESOLUCIÓN ESPACIAL DEL HEXAGONO
                    geo_json_conformant = True)
polygonise = lambda hex_id: Polygon(
                                  h3.h3_to_geo_boundary(
                                      hex_id, geo_json=True)
                                    )
all_polys = gpd.GeoSeries(list(map(polygonise, hexs)), \
                                      index = hexs, \
                                      crs = "EPSG:4326" \
                                      )

HEX_CDMX = gpd.GeoDataFrame(gpd.GeoSeries(all_polys))

HEX_CDMX = HEX_CDMX.rename(columns={0:'geometry'}).set_geometry('geometry').to_crs(6372)

HEX_CDMX.index.name = 'hex'

HEX_CDMX = HEX_CDMX.reset_index()

#############################################################################
##### GEOMETRIA DE MANZANAS DE LA CIUDAD DE MÉXICO


MANZA_CDMX = MANZA_CDMX[['CVEGEO',
                         'POBTOT',
                         'VIVTOT',
                         'geometry']].to_crs(6372)

#############################################################################
##### HOGARES CENSADO POR INEGI EN 2020

HOG = HOGAR.query('NOM_LOC!="Total de la entidad" & NOM_LOC!="Total del municipio" & NOM_LOC!="Total de la localidad urbana" & NOM_LOC!="Total AGEB urbana"')

HOG = HOG[[ 'ENTIDAD', 'NOM_ENT', 'MUN', 'NOM_MUN',
            'LOC','NOM_LOC', 'AGEB', 'MZA',
            'POBTOT', 'TOTHOG']]

HOG = HOG.replace({'*': np.nan, 'N/D': np.nan})

HOG['TOTHOG'] = HOG['TOTHOG'].astype(float)

Col_str = ['ENTIDAD', 'NOM_ENT', 'MUN',
           'NOM_MUN', 'LOC', 'NOM_LOC',
           'AGEB', 'MZA']

HOG[Col_str] = HOG[Col_str].astype(str)

HOG['ENTIDAD'] = HOG['ENTIDAD'].str.zfill(2)
HOG['MUN'] = HOG['MUN'].str.zfill(3)
HOG['LOC'] = HOG['LOC'].str.zfill(4)
HOG['MZA'] = HOG['MZA'].str.zfill(3)

HOG['CVEGEO'] = HOG[['ENTIDAD', 'MUN',
                     'LOC','AGEB','MZA']].agg(''.join, axis=1)

HOGARES = MANZA_CDMX.merge( HOG,
                            left_on = 'CVEGEO',
                            right_on = 'CVEGEO',
                            how = 'left')

HOGARE = HOGARES[['CVEGEO','geometry']]

#############################################################################
#####  HOGARES EN HEXAGONOS

Intersection = gpd.overlay( HEX_CDMX,
                            HOGARE,
                            how = 'intersection')

Intersection['area'] = Intersection.geometry.area

Intersection = ( Intersection.sort_values('area').
                 drop_duplicates(Intersection.columns, keep = 'last').
                 drop(columns = 'geometry'))

Inter = Intersection[['hex', 'CVEGEO']]
HOGAR = HOGARES[['CVEGEO', 'TOTHOG']]

HOGARES_ALL = Inter.merge( HOGAR,
                           left_on = 'CVEGEO',
                           right_on = 'CVEGEO',
                           how = 'left')

HOGARES_ALL = HOGARES_ALL.groupby('hex')['TOTHOG'].sum().reset_index()

HOGARES_ALL['geometry'] = HOGARES_ALL['hex'].apply(id2poly)

HOGARES_ALL = gpd.GeoDataFrame( HOGARES_ALL,
                                geometry = HOGARES_ALL['geometry'])

HOGARES_ALL.crs = "EPSG:4326"

HOGARES_ALL = HOGARES_ALL.to_crs("EPSG:4326")

#############################################################################
##### CONSTRUCCIÓN GEOESPACIAL: MAPA

HOGARES_ALL.plot( column = 'TOTHOG', figsize = (10,10))

#############################################################################
##### EXPORTACIÓN DE DATOS: SHP

HOGARES_ALL.to_file(Salidas + 'HOGARES_ALL.shp')
