#############################################################################
##### Accesibility
# This file produces an accesibility model for servicers defined in file OFERTA
#############################################################################

#### Imports
from h3 import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from access import Access, weights, Datasets

import warnings
warnings.filterwarnings('ignore')

def id2poly(hex_id):
    boundary = [x[::-1] for x in h3.h3_to_geo_boundary(hex_id)]
    return Polygon(boundary)

#############################################################################
##### Inputs

Entrada = '../data'
Salida = '../data'

#############################################################################
##### Base hexagons

HEX_CDMX = gpd.read_file(Entrada + 'HEX_CDMX.shp')

HOGARES_ALL = gpd.read_file(Entrada + 'HOGARES_ALL.shp')

OFERTA = gpd.read_file(Entrada + 'office.shp')

#############################################################################
##### Join services to hexagons

Hex_Unidad = gpd.sjoin( HEX_CDMX,
                        OFERTA,
                        how = "right",
                        op = 'contains')

#############################################################################
##### Count services by hexagon

Hex_Unidad = Hex_Unidad.groupby('hex')['name'].count().reset_index()
Hex_Unidad ['geometry'] = Hex_Unidad['hex'].apply(id2poly)
Hex_Unidad = gpd.GeoDataFrame( Hex_Unidad,
                               geometry = Hex_Unidad['geometry'])
Hex_Unidad.crs = "EPSG:4326"
Hex_Unidad = Hex_Unidad.to_crs("EPSG:4326")
Hex_Unidad = Hex_Unidad.to_crs(6372)

#############################################################################
##### SPATIAL ACCESIBILITY : GRAVITY MODEL

Unidad_access = Access( demand_df = HOGARES_ALL,
                        demand_index = "hex",
                        demand_value=  "TOTHOG",
                        supply_df = Hex_Unidad,
                        supply_index = "hex",
                        supply_value = "name")


Unidad_access.create_euclidean_distance( name = "euclidean_neighbors",
                                         threshold = 5000, ### Unidad Metros
                                         centroid_d = True,
                                         centroid_o=True)

gravity = weights.gravity( scale = 30,
                           alpha = -2,
                           min_dist = 10)

weighted = Unidad_access.weighted_catchment( name = "gravity",
                                             weight_fn = gravity)


#############################################################################
##### GRAVITY MODEL WEIGHTS

weighted = weighted.reset_index()
weighted ['geometry'] = weighted ['hex'].apply(id2poly)
weighted = gpd.GeoDataFrame( weighted,
                             geometry = weighted['geometry'])
weighted.crs = "EPSG:4326"
weighted = weighted.to_crs("EPSG:4326")
weighted = weighted.rename({'gravity_name':'gravity_value'}, axis=1)
weighted['gravity_value'] = weighted['gravity_value'].fillna(0)

print('Max value',weighted.gravity_value.max())
print('Min value',weighted.gravity_value.min())
print('Weighted shape',weighted.shape)

#############################################################################
##### PLOT

BASE = HOGARES_ALL.to_crs(4326).plot( color="black",
                                      markersize=7,
                                      figsize=(11, 11))

weighted.plot( ax = BASE,
               column = "gravity_value",
               scheme = 'Quantiles',
               cmap = 'viridis',
               classification_kwds = {'k':8})

OFERTA.to_crs(4326).plot( ax = BASE,
                          color = "red",
                          markersize = 1)

#############################################################################
##### Save as csv

weighted[[ 'hex', 'gravity_value']].to_csv(Salida + "Oficinas_SA.csv", index = False)

#############################################################################
