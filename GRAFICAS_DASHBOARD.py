#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%
#
# TABLERO PARA LAS SUPERVICIONES - GRAFICOS
#
#> Autor: Ladino Álvarez Ricardo Arturo
#> Área: CECTI


#%%%%%%%%%%%%%%%% Librerias de uso base %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore",
                        category = DeprecationWarning)



#%%%%%%%%%%%%%%%% Graficos del menu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> Gráfica de Mapa-Menu
def MAPA_MENU(Tabla, GeoJson):
    """Genera un mapa interactivo basado en una tabla y un GeoJSON."""
    UNIDADES = Tabla.iloc[:, 0].unique()
    GeoJson['color'] = GeoJson['UNIDAD_ADM'].apply(lambda x: 'red' if x in UNIDADES else 'grey')
   
    # Filtrar geometrías de las regiones seleccionadas
    geometries = GeoJson[GeoJson['UNIDAD_ADM'].isin(UNIDADES)]['geometry']

    # Calcular los límites combinados de las regiones seleccionadas
    if not geometries.empty:
        combined_bounds = geometries.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (combined_bounds[1] + combined_bounds[3]) / 2
        center_lon = (combined_bounds[0] + combined_bounds[2]) / 2

        # Ajustar el nivel de zoom basado en la extensión
        zoom = 8  # Valor por defecto
        extent_lat = combined_bounds[3] - combined_bounds[1]
        extent_lon = combined_bounds[2] - combined_bounds[0]
        max_extent = max(extent_lat, extent_lon)
        if max_extent > 0:
            zoom = max(3, min(10, 12 - max_extent))  # Zoom dinámico según la extensión
    else:
        # Si no hay regiones seleccionadas, usar valores predeterminados
        center_lat, center_lon, zoom = 23.798375, -102.5821213, 3.5 
    
    # Configuración del mapa
    MAPA = px.choropleth_mapbox(GeoJson,
                                geojson = GeoJson['geometry'],
                                locations = GeoJson.index,
                                color = 'color',
                                color_discrete_map = {'red': 'rgb(99,13,50)', 'grey': 'rgb(111, 111, 113)'},
                                mapbox_style = "open-street-map",
                                center = {'lat': center_lat, 'lon': center_lon},
                                zoom = 3.5,
                                opacity = 0.5,
                                )

    MAPA.update_traces(marker_line_width = 0.5, hovertemplate = None, hoverinfo = 'none')
    
    # Actualización de trazos y diseño
    MAPA.update_layout(showlegend = False,
                       title = dict(text = "Unidad Administrativa",
                                    x = 0.5,
                                    xanchor = 'center',
                                    font = dict(family = "Geomanist Light", size = 16)
                                    ),
                       font = dict(family = "Geomanist Light"),
                       margin = dict(r = 0, t = 55, l = 0, b = 55)
                       )
    
    return (MAPA)

#%%%%%%
#>  Gráfica de Dona-Menu

def DONA_MENU(Tabla, Variable):
    """Genera una gráfica de dona para mostrar puestos con y sin riesgo."""
    Labels = ['Puesto con Riesgo', 'Puesto sin Riesgo']
    Total = Tabla.PUESTO_NOM.nunique()
    Values = [Tabla[Tabla[Variable] == 'SI'].PUESTO_NOM.nunique(),
              Tabla[Tabla[Variable] == 'NO'].PUESTO_NOM.nunique()
             ]

    PASTEL = go.Figure(data = [go.Pie(labels = Labels,
                                      values = Values,
                                      hole = 0.5,
                                      textinfo = 'label+value+percent',
                                      hoverinfo = 'label+value+percent',
                                      texttemplate = '%{label}<br>%{value} (%{percent})',
                                      textposition = 'outside',
                                      marker=dict(colors = ['rgb(99,13,50)', 'rgb(111, 111, 113)'],
                                                  line = dict(color = '#FFFFFF', width = 0.5)
                                                  )
                                      )
                               ]
                      )

    # Configuración general de la gráfica
    PASTEL.update_layout(annotations=[dict(text = f'Total Puestos<br>{Total}',
                                           x = 0.5, y = 0.5,
                                           font_size = 16,
                                           showarrow = False,
                                           font = dict(family = "Geomanist Light")
                                           )
                                     ],
                         showlegend = False,
                         title = dict(text = "PUESTOS EN RIESGO",
                                      x = 0.5,
                                      xanchor = 'center',
                                      font=dict(family = "Geomanist Light", size = 16)
                                      ),
                         margin = dict(r = 60, t = 60, l = 60, b = 60),
                         font = dict(family = "Geomanist Light", size = 13, color = '#000000'),
                        )

    # Configuración adicional de los trazos
    PASTEL.update_traces(marker_line_width = 0.5, hovertemplate = None, hoverinfo = 'none')

    return (PASTEL)



#%%%%%%
#>  Gráfica de Barras-Menu & Riesgos
def BARRAS_MENU(Tabla):
    """Genera una gráfica de barras para mostrar la distribución de empleados por puesto."""
    VSN = (Tabla[Tabla['PTO_RIESGO'] == 'SI']
           .groupby(['CVE_UNIDAD', 'UNIDAD', 'PUESTO_NOM'])['EMPLEADO']
           .nunique()
           .reset_index()
           .sort_values(by = ['EMPLEADO'], ascending = False))
    
    VSN['PUESTO_NOM_ORD'] = VSN['PUESTO_NOM'].astype(str) + ' '
    
    unique_positions = VSN['PUESTO_NOM_ORD'].unique()
    num_positions = len(unique_positions)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    colors = ([mcolors.to_hex(cmap(i / (num_positions - 1))) for i in range(num_positions)]
              if num_positions > 1 else [mcolors.to_hex(cmap(0))])
    
    color_map = dict(zip(unique_positions, colors))
    
    BARRAS = px.bar(VSN,
                    y = 'EMPLEADO',
                    x = 'PUESTO_NOM_ORD',
                    orientation = 'v',
                    color = 'PUESTO_NOM_ORD',
                    color_discrete_map = color_map,
                    text = 'EMPLEADO'
                    )
    
    # Configuración general de la gráfica
    BARRAS.update_traces(marker_line_width = 0.5,
                         textposition = 'auto',
                         hovertemplate = None,
                         hoverinfo = 'none'
                         )
    
    BARRAS.update_layout(plot_bgcolor = '#FFFFFF',
                         xaxis = dict(title = "",
                                      showticklabels = False,
                                      visible = True,
                                      categoryorder = 'total descending'
                                      ),
                         yaxis = dict(title = dict(text = "No. Empleados",
                                                   font = dict(size = 13, color="#000000") 
                                                   ),
                                      showticklabels=False,
                                      ticks = '',
                                      showgrid = False,
                                      visible = True
                                      ),
                         margin = dict(r = 0, t = 0, l = 0, b = 0),
                         legend = dict(title = dict(text = 'Puestos', font = dict(size = 13, color = "#000000")),
                                       orientation = "h",
                                       yanchor = "top",
                                       y = -0.1,
                                       xanchor = "center",
                                       x = 0.5,
                                       font = dict(size = 12),
                                       traceorder = "normal"
                                       ),
                         title = dict(text = "", 
                                      x = 0.5,
                                      font = dict(family = "Geomanist Light", size = 16, color = '#000000')
                                      ),
                         font = dict(family = "Geomanist Light",
                                     size = 14,
                                     color='#000000'
                                     )
                        )
    
    return (BARRAS)


#%%%%%%
#>  Gráfica de Confiabilidad-Riesgos
def CONFIABILIDAD(Tabla):
    """Genera una gráfica de funnel para datos de confiabilidad."""
    #> Agrupación de datos
    confiabilidad_data = (Tabla.groupby('PREVE_OBS')['EMPLEADO'].nunique().reset_index().sort_values(by = 'EMPLEADO', 
                                                                                                     ascending = False)
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32',
                                                          '#b18f5f', 
                                                          '#6f6f71'])
    num_classes = len(confiabilidad_data['EMPLEADO'])

    #> Configuración de colores
    if num_classes > 1:
        colors = [mcolors.to_hex(cmap(i/(num_classes -1))) for i in range(num_classes)]
    else:
        colors = [mcolors.to_hex(cmap(0))]

    #> Información de texto
    text_info = [f"{numero}<br>{texto}" for numero, texto in zip(confiabilidad_data['EMPLEADO'], 
                                                                 confiabilidad_data['PREVE_OBS'])]

    #> Creación de la gráfica
    funnel_confiabi = go.Figure(go.Funnel(y = confiabilidad_data['PREVE_OBS'],
                                          x = confiabilidad_data['EMPLEADO'],
                                          text = text_info,
                                          textposition = "auto",
                                          textinfo = "text",
                                          orientation = 'h',
                                          marker = {"color": colors},
                                          connector = {"line": {"color": "#000000", "dash": "solid", "width": 0.5},
                                                       "fillcolor": "#dee8e7"}
                                          )
                               )

    #> Configuración de diseño
    funnel_confiabi.update_layout(title = {'text': "CONFIABILIDAD",
                                           'x': 0.5,
                                           'xanchor': 'center',
                                           'font': {'family': "Geomanist Light", 'size': 14}
                                           },
                                  font = {'family': "Geomanist Light",
                                          'size': 12,
                                          'color': '#000000'
                                        },
                                  template = None,
                                  showlegend = False,
                                  margin = {"r": 50, "t": 50, "l": 50, "b": 50},
                                  yaxis = dict(title = dict(text = "No. Empleados",
                                                            font = dict(size = 13, color = "#000000") 
                                                            ),
                                               showticklabels = False,
                                               ticks = '',
                                               showgrid = False,
                                               visible = True
                                               ),
                                  )

    #> Configuración de trazos
    funnel_confiabi.update_traces(hovertemplate = None, hoverinfo = 'none')

    return (funnel_confiabi)


#%%%%%%
#>  Gráfica de Quejas & Denuncias-Riesgos
def QUEJA_DENUNCIA(Tabla):
    """Genera una gráfica de funnel para visualizar quejas y denuncias."""
    # Agrupación de datos
    queja_denuncia_data = (Tabla.groupby('ESTATUS')['EMPLEADO'].nunique().reset_index().sort_values(by = 'EMPLEADO',
                                                                                                    ascending = False))                                                                                             

    # Configuración de colores
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    num_classes = len(queja_denuncia_data['EMPLEADO'])

    #> Configuración de colores
    if num_classes > 1:
        colors = [mcolors.to_hex(cmap(i/(num_classes -1))) for i in range(num_classes)]
    else:
        colors = [mcolors.to_hex(cmap(0))]


    # Información de texto
    text_info = [f"{numero}<br>{texto}" for numero, texto in zip(queja_denuncia_data['EMPLEADO'], queja_denuncia_data['ESTATUS'])]

    # Creación de la gráfica
    funnel_qd = go.Figure(go.Funnel(y = queja_denuncia_data['ESTATUS'],
                                    x = queja_denuncia_data['EMPLEADO'],
                                    text = text_info,
                                    textposition = "auto",
                                    textinfo = "text",
                                    orientation = 'h',
                                    marker = {"color": colors},
                                    connector = {"line": {"color": "#000000", "dash": "solid", "width": 0.5},
                                                 "fillcolor": "#dee8e7"}
                                    )
                         )

    # Configuración de diseño
    funnel_qd.update_layout(title={'text': "QUEJAS Y DENUNCIAS",
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'font': {'family': "Geomanist Light", 'size': 14}
                                   },
                            font = {'family': "Geomanist Light",
                                    'size': 12,
                                    'color': '#000000'
                                   },
                            template = None,
                            showlegend = False,
                            margin = {"r": 50, "t": 50, "l": 50, "b": 50},
                            yaxis = dict(title = dict(text = "No. Empleados",
                                                      font = dict(size = 13, color = "#000000")),
                                         showticklabels = False,
                                         ticks = '',
                                         showgrid = False,
                                         visible = True
                                         ),
                            )

    # Configuración de trazos
    funnel_qd.update_traces(hovertemplate = None, hoverinfo = 'none')

    return (funnel_qd)


#%%%%%%
#>  Gráfica de Nivel Atención-Riesgos
def RIESGO_ATENCION_NIVEL(Tabla):
    """Genera una gráfica de Treemap alineando colores con el nivel de atención."""
    # Agrupación de datos
    nivel_riesgo = (Tabla.groupby('NIVEL_ATENCION')['EMPLEADO'].nunique().reset_index().sort_values(by =['EMPLEADO'], 
                                                                                                    ascending = False))

    # Diccionario de colores alineados con NIVEL_ATENCION
    nivel_colores = {'NULO': '#6f6f71',
                     'BAJO': '#907f68',
                     'MEDIO': '#b18f5f',
                     'ALTO': '#8a4e48',
                     'CRITICO': '#630d32'
                     }

    # Mapear colores a las clases de NIVEL_ATENCION
    colors = nivel_riesgo['NIVEL_ATENCION'].map(nivel_colores)

    # Creación de la gráfica de Treemap
    tree_atencion = go.Figure(go.Treemap(labels = nivel_riesgo['NIVEL_ATENCION'],
                                         parents = [""] * len(nivel_riesgo['NIVEL_ATENCION']),
                                         values = nivel_riesgo['EMPLEADO'],
                                         marker = dict(colors = colors),
                                         textinfo = 'label+value',
                                         textposition = 'middle center'
                                         )
                              )

    # Configuración de diseño
    tree_atencion.update_layout(title = "",
                                font = dict(
                                family = "Geomanist Light",
                                size = 12,
                                color = '#000000'),
                                template = None,
                                showlegend = False,
                                margin = {"r": 0, "t": 0, "l": 0, "b": 0},
                                uniformtext = dict(minsize = 14, mode = 'hide')
                                )

    # Configuración de trazos
    tree_atencion.update_traces(hovertemplate = None, hoverinfo = 'none')

    return (tree_atencion)


#%%%%%%
#>  Tabla de Detalle aplicativos - Riesgo
def APLICATIVOS_TABLA (Tabla):
    ''' '''
    Tabla.rename(columns = {'RFC_CORTO': 'RFC corto',
                            'EMPLEADO': 'No. Empleado', 
                            'NOMBRE_EMP': 'Nombre Empleado',
                            'PTO_RIESGO':'Puesto Riesgo (ACAER)',
                            'PREVE_OBS':'Confiabilidad Estatus',
                            'DENUNCIAS':'No. Denuncias',
                            'NIVEL_ATENCION':'Nivel Riesgo',
                            'APLICATIVO':'No. Sistemas'}, inplace = True)
    
    pto_riesgo_colors = ['#630d32' if val == 'SI' else '#6f6f71' for val in Tabla['Puesto Riesgo (ACAER)']]


    nivel_atencion_colors = []
    for val in Tabla['Nivel Riesgo']:
        if val == 'MEDIO':
            nivel_atencion_colors.append('#b18f5f')
        elif val == 'ALTO':
            nivel_atencion_colors.append('#8a4e48')
        elif val == 'CRITICO':
            nivel_atencion_colors.append('#630d32')
        else:
            nivel_atencion_colors.append('#FFFFFF')

    pto_riesgo_text_colors = ['#ffffff' if val == 'SI' else '#ffffff' for val in Tabla['Puesto Riesgo (ACAER)']]
    
    TABLA = go.Figure(data = [go.Table(header = dict(values = list(Tabla.columns),
                                                     fill_color = '#80776c',
                                                     font = dict(color = '#FFFFFF'), 
                                                     align='center'),
                                       cells=dict(values = [Tabla[col] for col in Tabla.columns],
                                                  fill_color = [['#FFFFFF']*len(Tabla),
                                                                ['#FFFFFF']*len(Tabla),
                                                                ['#FFFFFF']*len(Tabla), 
                                                                pto_riesgo_colors, 
                                                                ['#FFFFFF']*len(Tabla), 
                                                                ['#FFFFFF']*len(Tabla),
                                                                nivel_atencion_colors,
                                                                ['#FFFFFF']*len(Tabla)],
                                                  font = dict(color = [['#000000']*len(Tabla),
                                                                       ['#000000']*len(Tabla),
                                                                       ['#000000']*len(Tabla),
                                                                       pto_riesgo_text_colors,
                                                                       ['#000000']*len(Tabla),
                                                                       ['#000000']*len(Tabla),
                                                                       ['#ffffff']*len(Tabla),
                                                                       ['#000000']*len(Tabla)]
                                                             ),
                                                  line_color = '#000000',
                                                  align = 'center'))
                             ]
                     )
    
    TABLA.update_layout(title_font_size = 12,
                        title = {'text': "",
                                  'x': 0.5, 
                                  'xanchor': 'center', 
                                  'font': {'family': "Geomanist Light",
                                  'size': 12}
                                },
                        font = dict(family = "Geomanist Light",
                                    size = 12,
                                    color = '#000000'),
                        template = None, showlegend = False,
                        margin = {"r": 10, "t": 50, "l": 10, "b": 50},
                        uniformtext=dict(minsize = 10, mode ='hide')
                       )
    
    return (TABLA)



#%%%%%%
#>  Gráfica de Accesos - Riesgo
def ACCESOS_APP(Tabla):
    ''' '''
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    
    num_apps = len(Tabla)

    #> Configuración de colores
    if num_apps > 1:
        colors = [mcolors.to_hex(cmap(i/(num_apps -1))) for i in range(num_apps)]
    else:
        colors = [mcolors.to_hex(cmap(0))]
    

    max_value = Tabla['ROL_APP'].max()
    
    pull_values = [0.05 if val > 0.80 * max_value else 0 for val in Tabla['ROL_APP']]
    
    PASTEL = px.pie(Tabla,
                    names = 'APLICATIVO',
                    values = 'ROL_APP')
    
    PASTEL.update_traces(pull = pull_values,
                         marker = dict(colors = colors,
                                       line = dict(color = '#FFFFFF', width = 0.5)
                                      ),
                         textposition = 'outside',
                         textinfo = 'label+percent', 
                         insidetextorientation = 'radial',
                         hovertemplate = None,
                         hoverinfo =  'none')
    
    PASTEL.update_traces(pull = pull_values,
                         marker = dict(colors = colors), 
                         hole=0
                        )
    
    PASTEL.update_layout(title_font_size = 10,
                         title = {'text': "",
                                  'x': 0.5, 
                                  'xanchor': 'center', 
                                  'font': {'family': "Geomanist Light",
                                  'size': 12}
                                },
                         font = dict(family = "Geomanist Light",
                                     size = 11,
                                     color = '#000000'),
                         template = None, showlegend = False,
                         margin = {"r": 10, "t": 10, "l": 10, "b": 10},
                         uniformtext = dict(minsize = 5, mode ='hide')
                        )
    return (PASTEL)

    
#%%%%%%
#>  Gráfica de Aplicativos - Aplicativos
def APLICATIVOS_TOTAL(Tabla):
    """Genera una gráfica de barras para mostrar el total de aplicativos por desconcentrada."""
    TOT_APP = Tabla.groupby('DESCONCENTRADA')['APLICATIVO'].nunique().reset_index().sort_values(by = ['APLICATIVO'],
                                                                                                ascending=False)
    TOT_APP['DESCONCENTRADA_ORD'] = TOT_APP['DESCONCENTRADA'].astype(str) + ' '
    unique_positions = TOT_APP['DESCONCENTRADA_ORD'].unique()
    num_positions = len(unique_positions)

    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32','#b18f5f', '#6f6f71'])

    if num_positions > 1:
        colors = [mcolors.to_hex(cmap(i/(num_positions -1))) for i in range(num_positions)]
    else:
        colors = [mcolors.to_hex(cmap(0))]

    color_map = dict(zip(unique_positions, colors))
    
    TOTAL_APP = px.bar(TOT_APP,
                       y = 'APLICATIVO',
                       x = 'DESCONCENTRADA_ORD',
                       orientation = 'v',
                       color = 'DESCONCENTRADA_ORD',
                       color_discrete_map = color_map,
                       text = 'APLICATIVO')
    
    TOTAL_APP.update_traces(marker_line_width = 0.5,
                            textposition = 'auto',
                            hovertemplate = None, 
                            hoverinfo = 'none')
    TOTAL_APP.update_layout(plot_bgcolor = 'white',
                            yaxis = dict(title = dict(text = "No. Aplicativos",
                                                      font = dict(size = 13, color = "#000000")),
                                         showticklabels = False,
                                         ticks = '',
                                         showgrid = False,
                                         visible = True
                                         ),
                            xaxis = dict(showticklabels = False,
                                         visible = False,
                                         categoryorder = 'total descending'
                                         ),
                            margin = dict(r = 10, t = 10, l = 10, b = 10),
                            legend = dict(title = dict(text = 'Administración', font = dict(size = 12, color = "#000000")),
                                          orientation = "h",
                                          yanchor = "top",
                                          y = -0.1,
                                          xanchor = "center",
                                          x = 0.5,
                                          font = dict(size = 12),
                                          traceorder = "normal"
                                          ),
                            title = dict(text = "",
                                       x = 0.5,
                                       font=dict(family = "Geomanist Light", size = 16, color = '#000000')
                                       ),
                            font = dict(family = "Geomanist Light", size = 16, color = '#000000')
                            )

    return (TOTAL_APP)


#%%%%%%
#>  Gráfica de Aplicativos & Puestos - Aplicativos
def APLICATIVOS_PUESTOS (Tabla):
    ''' '''
    TOT_APP = Tabla.groupby('PUESTO_NOM')['APLICATIVO'].nunique().reset_index().sort_values(by=['APLICATIVO'], 
                                                                                            ascending=False)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    num_apps = len(TOT_APP)
    colors = [mcolors.to_hex(cmap(i / (num_apps - 1))) for i in range(num_apps)]
    max_value = TOT_APP['APLICATIVO'].max()
    pull_values = [0.05 if val > 0.9 * max_value else 0 for val in TOT_APP['APLICATIVO']]

    PASTEL_APP = px.pie(TOT_APP,
                        names = 'PUESTO_NOM',
                        values = 'APLICATIVO')
    
    PASTEL_APP.update_traces(pull = pull_values,
                             marker = dict(colors = colors,
                                           line = dict(color = '#FFFFFF',
                                                       width = 0.5)),
                             textposition ='outside',
                             textinfo ='label+percent', 
                             insidetextorientation = 'radial',
                             hovertemplate = None, 
                             hoverinfo= 'none'
                            )
    PASTEL_APP.update_traces(pull = pull_values, marker=dict(colors=colors), hole=0)
    PASTEL_APP.update_layout(title ="", title_font_size = 12,
                             title_x = 0.5,
                             font = dict( family = "Geomanist Light",
                                         size = 10,
                                         color = '#000000'),
                             template = None,
                             showlegend = False,
                             margin = {"r": 0, "t": 0, "l": 0, "b": 30},
                             uniformtext = dict(minsize = 10, mode ='hide')
                            )
    return (PASTEL_APP)


#%%%%%%
#>  Grafica Puestos & Sistemas - Aplicativos
def MAYOR_SISTEMAS(Tabla):
    ''' '''
    PUESTO_SISTEMAS = Tabla.groupby(['DESCONCENTRADA',
                                      'PUESTO_NOM'])['APLICATIVO'].nunique().reset_index().sort_values(by = ['APLICATIVO'],
                                                                                                       ascending = False)
    PUESTO_SISTEMAS['PUESTO_NOM_ORD'] = PUESTO_SISTEMAS['PUESTO_NOM'].astype(str) + ' '
    unique_positions = PUESTO_SISTEMAS['PUESTO_NOM_ORD'].unique()
    num_positions = len(unique_positions)


    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])

    if num_positions > 1:
        colors = [mcolors.to_hex(cmap(i/(num_positions -1))) for i in range(num_positions)]
    else:
        colors = [mcolors.to_hex(cmap(0))]
    
    color_map = dict(zip(unique_positions, colors))
    
    BAR_PUESTOS = px.bar(PUESTO_SISTEMAS,
                         x = "APLICATIVO",
                         y = "PUESTO_NOM_ORD",
                         orientation = 'h',
                         text = PUESTO_SISTEMAS["APLICATIVO"],
                         color = 'PUESTO_NOM_ORD',
                         color_discrete_map = color_map,
                         title = ""
                        )
    BAR_PUESTOS.update_layout(yaxis = dict(side = 'right'),
                              xaxis = dict(autorange = 'reversed')
                             )
    BAR_PUESTOS.update_traces(marker_line_width = 0.5,
                              textposition = 'auto',
                              hoverinfo = 'none')
    BAR_PUESTOS.update_layout(plot_bgcolor = 'white', 
                              xaxis_title = "",
                              yaxis_title = "No. Aplicativos",
                              yaxis = dict(title = dict(text = "No. Aplicativos",
                                                        font = dict(size = 13, color = "#000000")),
                                                        showticklabels = False), 
                              xaxis = dict(showticklabels = False,
                                           categoryorder = 'total descending'),
                              margin = {"r": 0, "t": 0, "l": 0, "b": 0}, 
                              legend_title_text = 'Puestos',
                              legend=dict(orientation = "h",
                                          yanchor = "top",
                                          y = -0.1,
                                          xanchor = "center",
                                          x = 0.5,
                                          title_font = dict(size = 12),
                                          font = dict(size = 12),
                                          traceorder = "normal"
                                         )
                             )
    BAR_PUESTOS.layout.legend.update(title = dict(text = 'Puestos',
                                                  font = dict(size = 13, color = "#000000"), 
                                                  side = 'top'))
    BAR_PUESTOS.update_layout(title_font_size = 12,
                              title_x = 0.5,
                              font = dict(family = "Geomanist Light",
                                          size = 12,
                                          color = '#000000')
                             )
    BAR_PUESTOS.update_traces(hovertemplate = None, hoverinfo = 'none')

    return (BAR_PUESTOS)


#%%%%%%
#>  Gráfica Mayor uso de Apps - Aplicativos
def APLICATIVOS_MAYOR(Tabla):
    '''  '''
    APLICATIVOS_USO = Tabla.groupby(['DESCONCENTRADA','APLICATIVO'])['EMPLEADO'].\
                      count().reset_index().sort_values(by=['EMPLEADO'], ascending=False).head(10)
    
    r = APLICATIVOS_USO.EMPLEADO.values
    labels =  APLICATIVOS_USO.APLICATIVO.values
    
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    num_classes = len(APLICATIVOS_USO.APLICATIVO)

    if num_classes > 1:
        colors = [mcolors.to_hex(cmap(i/(num_classes -1))) for i in range(num_classes)]
    else:
        colors = [mcolors.to_hex(cmap(0))]

    
    num_slices = len(r)
    theta = [(i + 0.5) * 360 / num_slices for i in range(num_slices)]
    width = [360 / num_slices for _ in range(num_slices)]
             
    barpolar_plots = [go.Barpolar(r = [r_value], 
                                  theta = [t], 
                                  width = [w], 
                                  name = n, 
                                  marker_color = [c],
                                  marker_line_color = 'white',
                                  marker_line_width = 2,
                                  hovertemplate = f'{n}<br>Total: {r_value}<extra></extra>'
                                 )
                      for r_value, t, w, n, c in zip(r, theta, width, labels, colors)
                     ]
    
    ROSE_USO = go.Figure(barpolar_plots)
    
    ROSE_USO.update_layout(template = None,
                           title = "",
                           polar = dict(radialaxis = dict(showgrid = True,
                                                          gridcolor = '#6f6f71',
                                                          gridwidth = 0.2,
                                                          griddash = 'dot',
                                                          showline = False,
                                                          range = [APLICATIVOS_USO.EMPLEADO.values.min(), 
                                                                   APLICATIVOS_USO.EMPLEADO.values.max()],
                                                          showticklabels = True,
                                                          tickfont = dict(color = 'Black')),
                                        angularaxis=dict(showgrid = True,
                                                         gridcolor = 'gray',
                                                         gridwidth = 1,
                                                         griddash = 'dot',
                                                         showline = False,
                                                         showticklabels = False,
                                                         ticks = '')
                                       ),
                           hoverlabel = dict(
                               font=dict(family = "Geomanist Light",
                                         size = 12,
                                         color = "white")
                           )
                          )
    
    ROSE_USO.layout.legend.update(title=dict(text='Aplicativos por Puestos',
                                             font=dict(size=12),
                                             side='top')
                                 )
    
    ROSE_USO.update_layout(title_font_size = 12,
                           title_x = 0.5,
                           font=dict(family = "Geomanist Light",
                                     size = 16,
                                     color = '#000000'),
                           legend=dict(orientation = "h",
                                       yanchor = "auto",
                                       y = -0.2,
                                       xanchor = "center",
                                       x = 0.5,
                                       title_font = dict(size = 13, color = "#000000"),
                                       font = dict(size = 12),
                                       traceorder = "normal"),
                           width = 1000,
                           height = 600,
                           margin = {"r": 0, "t": 0, "l": 0, "b": 0}
                          )
    return (ROSE_USO)


#%%%%%%
#>  Tabla Servicios App - Aplicativos
def TABLA_APPS(Tabla):
    ''' '''
    def renombrar_columnas(tabla, nuevos_nombres):
        tabla.columns = nuevos_nombres
        return tabla
    
    nuevos_nombres = ['Nombre Empleado', 'RFC Corto', 'Puesto', 'Aplicativo', 'ROL', 'Alcance ROL']

    TB = Tabla.groupby(['NOMBRE_EMP','RFC_CORTO','PUESTO_NOM',
                        'APLICATIVO', 'ROL_APP', 'ALCANCE_ROL'])['PTO_RIESGO'].nunique()\
                      .reset_index().drop(['PTO_RIESGO'], axis = 1)

    TB = renombrar_columnas(TB, nuevos_nombres)

    grouped = TB.groupby('Nombre Empleado').agg(list).reset_index()

    nombre_emp = []
    rfc_corto = []
    puesto_nom = []
    aplicativo = []
    rol_app = []
    alcance_rol = []
    rowspans = []

    for i, row in grouped.iterrows():
        num_roles = len(row['Puesto'])
        nombre_emp.extend([row['Nombre Empleado']] + [''] * (num_roles - 1))
        rfc_corto.extend([row['RFC Corto'][0]] + [''] * (num_roles - 1))
        puesto_nom.extend(row['Puesto'])
        aplicativo.extend(row['Aplicativo'])
        rol_app.extend(row['ROL'])
        alcance_rol.extend(row['Alcance ROL'])
        rowspans.append(num_roles)

    Tabla_G = go.Figure(data = [go.Table(header = dict(values = list(TB.columns),
                                                    fill_color = '#80776c',
                                                    font = dict(color = '#FFFFFF'),
                                                    align = 'center'),
                                        cells=dict(values = [nombre_emp, rfc_corto, 
                                                             puesto_nom, aplicativo, 
                                                             rol_app, alcance_rol],
                                                   fill_color = '#FFFFFF', 
                                                   align = 'left',
                                                   line_color = '#000000')
                                       )
                               ]
                        )
    Tabla_G.update_layout(title_font_size=12,
                           title = {'text': "",
                                     'x': 0.5, 
                                     'xanchor': 'center', 
                                     'font': {'family': "Geomanist Light",
                                     'size': 12}
                                },
                           font=dict(family = "Geomanist Light",
                                     size = 12,
                                     color = '#000000'),
                           template = None, 
                           showlegend = False,
                           margin = {"r": 10, "t": 30, "l": 10, "b": 0},
                           uniformtext = dict(minsize = 10, 
                                              mode = 'hide')
                          )

    # Crear un DataFrame para exportar
    Tabla_E = pd.DataFrame({'Nombre Empleado': nombre_emp,
                            'RFC corto': rfc_corto,
                            'Puesto': puesto_nom,
                            'Aplicativo': aplicativo,
                            'ROL': rol_app,
                            'Alcance ROL': alcance_rol})

    return (Tabla_G, Tabla_E)


#%%%%%%
#>  Tipo de clase - Denuncias
def VEL_DENU(Tabla):
    """Genera velocímetros distribuidos dinámicamente en filas y columnas."""
    Veloc = go.Figure()

    # Número de columnas por fila
    cols = 3
    num_elements = len(Tabla)  # Número de elementos
    rows = (num_elements + cols - 1) // cols  # Calcular el número de filas necesarias

    # Crear los velocímetros
    for i, row in Tabla.iterrows():
        # Calcular el valor máximo dinámico basado en la suma de todas las clases del velocímetro
        max_value = Tabla[Tabla['D15'] == row['D15']]['VALUE'].sum() * 1.1

        Veloc.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=row['VALUE'],
                title={"text": row['D15'], "font": {"size": 13}},  # Reducir el tamaño de la letra aquí
                gauge={
                    'axis': {'range': [0, max_value]},
                    'bar': {'color': "#630d32"},
                    'steps': [{'range': [0, max_value], 'color': "#F1F1E8"}],
                    'threshold': {
                        'line': {'color': "#000000", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                },
                domain={'row': i // cols, 'column': i % cols}
            )
        )

    # Configurar diseño del gráfico
    Veloc.update_layout(title = {'text': ""},
        grid={'rows': rows, 'columns': cols, 'pattern': "independent"},
        title_font_size=5,
        font=dict(
            family="Montserrat",
            size=12,
            color='#000000'
        ),
        margin=dict(t=40, l=10, r=10, b=10)  # Ajustar márgenes para estética
    )

    return (Veloc)


#%%%%%%
#>  Puestos con más folios - Denuncias
def DENUNCIAS_PUESTOS(Tabla):
    ''' '''
    Tabla['D8_O'] = Tabla['D8'].astype(str) + ' '
    unique_positions = Tabla['D8_O'].unique()
    num_positions = len(unique_positions)

    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32','#b18f5f', '#6f6f71'])

    if num_positions > 1:
        colors = [mcolors.to_hex(cmap(i/(num_positions -1))) for i in range(num_positions)]
    else:
        colors = [mcolors.to_hex(cmap(0))]

    color_map = dict(zip(unique_positions, colors))
    
    BARRAS = px.bar(Tabla,
                    y = 'VALUE',
                    x = 'D8_O',
                    orientation = 'v',
                    color = 'D8_O',
                    color_discrete_map = color_map,
                    text = 'VALUE'
                    )
    
    # Configuración general de la gráfica
    BARRAS.update_traces(marker_line_width = 0.5,
                         textposition = 'auto',
                         hovertemplate = None,
                         hoverinfo = 'none'
                         )
    
    BARRAS.update_layout(plot_bgcolor = '#FFFFFF',
                         xaxis = dict(title = "",
                                      showticklabels = False,
                                      visible = True,
                                      categoryorder = 'total descending'
                                      ),
                         yaxis = dict(title = dict(text = "No. Folios",
                                                   font = dict(size = 13, color="#000000") 
                                                   ),
                                      showticklabels=False,
                                      ticks = '',
                                      showgrid = False,
                                      visible = True
                                      ),
                         margin = dict(r = 0, t = 0, l = 0, b = 0),
                         legend = dict(title = dict(text = 'Puestos', font = dict(size = 13, color = "#000000")),
                                       orientation = "h",
                                       yanchor = "top",
                                       y = -0.1,
                                       xanchor = "center",
                                       x = 0.5,
                                       font = dict(size = 12),
                                       traceorder = "normal"
                                       ),
                         title = dict(text = "", 
                                      x = 0.5,
                                      font = dict(family = "Geomanist Light", size = 16, color = '#000000')
                                      ),
                         font = dict(family = "Geomanist Light",
                                     size = 14,
                                     color='#000000'
                                     )
                        )
    return (BARRAS)


#%%%%%%
#>  Gráfica de Aplicativos & Puestos - Denuncias
def CLASES_DENUNCIAS (Tabla):
    ''' '''
    cmap = mcolors.LinearSegmentedColormap.from_list("", ['#630d32', '#b18f5f', '#6f6f71'])
    num_apps = len(Tabla)
    colors = [mcolors.to_hex(cmap(i / (num_apps - 1))) for i in range(num_apps)]
    max_value = Tabla['VALUE'].max()
    pull_values = [0.05 if val > 0.9 * max_value else 0 for val in Tabla['VALUE']]

    PASTEL_APP = px.pie(Tabla,
                        names = 'D14',
                        values = 'VALUE')
    
    PASTEL_APP.update_traces(pull = pull_values,
                             marker = dict(colors = colors,
                                           line = dict(color = '#FFFFFF',
                                                       width = 0.5)),
                             textposition ='auto',
                             textinfo ='label+percent', 
                             insidetextorientation = 'radial',
                             hovertemplate = None, 
                             hoverinfo= 'none'
                            )
    PASTEL_APP.update_traces(pull = pull_values, marker=dict(colors=colors), hole = 0.01)
    PASTEL_APP.update_layout(title = '', title_font_size = 12,
                             title_x = 0.5,
                             font = dict( family = "Geomanist Light",
                                         size = 12,
                                         color = '#000000'),
                             template = None,
                             showlegend = False,
                             margin = {"r": 10, "t": 10, "l": 10, "b": 30},
                             uniformtext = dict(minsize = 10, mode ='hide')
                            )
    return (PASTEL_APP)


#%%%%%%
#>  Tabla denuncias - Denuncias
def TABLA_DENU(Tabla):
    Tabla = Tabla.applymap(lambda x: x.replace('_x000D_', '') if isinstance(x, str) else x)
    def renombrar_columnas(tabla, nuevos_nombres):
        tabla.columns = nuevos_nombres
        return tabla
    
    nuevos_nombres = ['No. Empleado', 'RFC Corto', 'Nombre', 'Clave Unidad', 'Unidad','General', 'Clave Puesto',
                     'Puesto', 'PREVE', 'Folio SIDEQUS', 'Año', 'Expediente', 'Medio de Recepción',
                     'Clasificación del asunto','Tipo Asunto','Fecha que ocurrieron los hechos','Administración General donde ocurrieron los  hechos',
                      'Unidad administrativa', 'Fecha de recepción del asunto', 'Estatus del asunto',
                      'Descripción de los Hechos', 'Conducta Corrupción']

    TB = renombrar_columnas(Tabla, nuevos_nombres)
    Expo = pd.DataFrame(TB)
    return (Expo)