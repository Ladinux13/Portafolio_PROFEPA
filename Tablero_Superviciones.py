#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%
#
# TABLERO PARA LAS SUPERVICIONES - RIESGOS APLICATIVOS
#
#> Autor: Ladino Álvarez Ricardo Arturo
#> Área: CECTI


#%%%%%%%%%%%%%%%% Librerias de uso base %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import hmac
import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st
import geopandas as gpd

#%%%%%%%%%%%%%%%% Funciones de los gráficos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from GRAFICAS_DASHBOARD import MAPA_MENU, DONA_MENU, BARRAS_MENU
from GRAFICAS_DASHBOARD import CONFIABILIDAD, QUEJA_DENUNCIA, RIESGO_ATENCION_NIVEL
from GRAFICAS_DASHBOARD import APLICATIVOS_TABLA, ACCESOS_APP
from GRAFICAS_DASHBOARD import APLICATIVOS_TOTAL, APLICATIVOS_PUESTOS, MAYOR_SISTEMAS
from GRAFICAS_DASHBOARD import APLICATIVOS_MAYOR, TABLA_APPS
from GRAFICAS_DASHBOARD import VEL_DENU, DENUNCIAS_PUESTOS, CLASES_DENUNCIAS
from GRAFICAS_DASHBOARD import TABLA_DENU

#%%%%%%%%%%%%%%%% Funciones de los contenedores %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 

from CONTENEDORES_DASHBOARD import Contenedor_Titulo, Contenedor_Hacienda, Contenedor_SAT
from CONTENEDORES_DASHBOARD import CONTENEDOR_M, CONTENEDOR_PR, CONTENEDOR_QDNA
from CONTENEDORES_DASHBOARD import ROLES_EMPLEADO, APLICATIVOS_RIESGOS, APP_SISTE_MAX
from CONTENEDORES_DASHBOARD import DENUNCIAS_CONTE, DENUN_PTO, CONTE_CLASS_SUNTO


#%%%%%%%%%%%%%%%% Funciones varias %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 

from FUNCIONES_VARIAS import Base_Datos, Base_Mapa, Puestos_Metricas
from FUNCIONES_VARIAS import APLICATIVOS_DETALLES, EXP_TABLA
from FUNCIONES_VARIAS import DENUNCIAS_INFO



#%%%%%%%%%%%%%%%% Mensajes de error %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import warnings
warnings.filterwarnings("ignore",
                        category = DeprecationWarning)

project_dir = os.path.dirname(os.path.abspath(__file__))


#%%%%%%%%%%%%%%%% Entrada de información %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> Bases de datos : Riesgos & Denuncias
Entradas = 'C:/Users/LAAR8976/Ladino_ALL/CECTI/DASHBOARD_RIESGOS_CECTI/BASE_SALIDA/'

#> CSS de Bootstrap
bootstrap_css_path = "C:/Users/LAAR8976/Ladino_ALL/CECTI/DASHBOARD_RIESGOS_CECTI/TABLERO/.css/bootstrap.min.css"


#%%%%%%%%%%%%%%%% Configuración de la página %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> Contraseña
def check_password():

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    st.text_input(
        "CONTRASEÑA", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("CONTRASEÑA INCORRECTA")
    return False

#if not check_password():
 #   st.stop() 


#> Titulo de la pagina
st.set_page_config(page_title = 'Análisis del Riesgo', 
                   layout = 'wide',
                   page_icon = r"C:/Users/LAAR8976/Ladino_ALL/CECTI/DASHBOARD_RIESGOS_CECTI/TABLERO/.image/Logo.jpg")


#> Leer el contenido del archivo CSS
with open(bootstrap_css_path, "r") as f:
    bootstrap_css_content = f.read()

#> Fuente personalizada Geomanist
font_css = f"""
<style>
{bootstrap_css_content}

/* Personalización de la fuente Geomanist */
@font-face {{
    font-family: 'Geomanist';
    src: url('file://{project_dir}/.fonts/Geomanist-Light.ttf') format('truetype');
    font-weight: normal;
}}

@font-face {{
    font-family: 'Geomanist';
    src: url('file://{project_dir}/.fonts/Geomanist-Light.ttf') format('truetype');
    font-weight: normal;
}}

body {{
    font-family: 'Geomanist' !important;
}}

h1, h2, h3, h4, h5, h6, p, div {{
    font-family: 'Geomanist' !important;
}}

</style>
"""

#> Creacion de CSS
#> CSS de cuadro texto
st.markdown(font_css, unsafe_allow_html=True)

css_succes = """
<style>
div.stAlert {
    background-color: #630d32;
    text-align: center;
}
div.stAlert p {
    color: #FFFFFF;
    font-weight: bold;
}
</style>
"""

#> CSS de boton
st.markdown(css_succes, unsafe_allow_html=True)

st.markdown("""
        <style>
        .stDownloadButton button {
            background-color: #6f6f71;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

#> CSS para reducir el margen superior
st.markdown(
    """
    <style>
    /* Reducir el margen superior del contenido de la aplicación */
    .main .block-container {
        padding-top: 0rem; /* Ajusta este valor según sea necesario */

    }
    </style>
    """,
    unsafe_allow_html=True
)


#%%%%%%%%%%%%%%%% Configuración de título %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> Titulo e imagenes de la pagina
st.markdown(css_succes, unsafe_allow_html = True)
Hacienda, Titulo, SAT = st.columns([2.5,5.5,2.5])
Contenedor_Hacienda(Hacienda, r'C:/Users/LAAR8976/Ladino_ALL/CECTI/DASHBOARD_RIESGOS_CECTI/TABLERO/.image/Hacienda.jpg')
Contenedor_Titulo(Titulo)
Contenedor_SAT(SAT, r'C:/Users/LAAR8976/Ladino_ALL/CECTI/DASHBOARD_RIESGOS_CECTI/TABLERO/.image/SAT.jpg')



#%%%%%%%%%%%%%%%% Bases de datos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BASE_RIESGO = Base_Datos(Entradas + "RIESGO.pkl")
BASE_DENUNCIAS = Base_Datos(Entradas + "DENUNCIAS.pkl")
BASE_MAPA = Base_Mapa(Entradas + "MAP_UAD.json")


#%%%%%%%%%%%%%%%% Barra lateral - Filtro UAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#>
st.sidebar.header("UNIDADES ADMINISTRATIVAS")

#> Crear la lista de unidades administrativas únicas a partir de ambas tablas
UAD_1_RIESGO = BASE_RIESGO["UNIDAD"].unique().tolist()
UAD_1_DENUNCIAS = BASE_DENUNCIAS["UNIDAD"].unique().tolist()

#> Unir las listas de unidades y eliminar duplicados
UAD_1 = sorted(list(set(UAD_1_RIESGO + UAD_1_DENUNCIAS)))

#> Crear el selector en la barra lateral
UAD_DROP = st.sidebar.multiselect("Seleccionar por Unidad:",
                                   options=UAD_1,
                                   default=[],  # Ninguna seleccionada por defecto
                                   key='UAD_DROP'
                                   )

#> Inicializar las variables para que estén accesibles fuera del bloque
BASE_RIESGOS_UNIDAD = pd.DataFrame()
BASE_DENUNCIAS_UNIDAD = pd.DataFrame()

#> Verificar si se seleccionaron unidades
if not UAD_DROP:
    st.sidebar.warning("⚠️ No se han seleccionado datos")
else:
    #> Filtrar los DataFrames
    BASE_RIESGOS_UNIDAD = BASE_RIESGO[BASE_RIESGO["UNIDAD"].isin(UAD_DROP)]
    BASE_DENUNCIAS_UNIDAD = BASE_DENUNCIAS[BASE_DENUNCIAS["UNIDAD"].isin(UAD_DROP)]


#> Espacio adicional para estética
st.sidebar.write("") 
st.sidebar.write("")  
st.sidebar.write("")
st.sidebar.write("") 
st.sidebar.write("") 
st.sidebar.write("") 
st.sidebar.write("")
st.sidebar.write("") 
st.sidebar.write("") 
st.sidebar.write("") 

#> Botón para borrar el caché en la parte inferior de la barra lateral
with st.sidebar:
    st.write("---")  #> Línea separadora opcional
    if st.button("Actualizar información"):
        st.cache_data.clear()  # Borra el caché de todas las funciones decoradas con @st.cache_data
        st.success("Información actualizada")                          


####################################################


## Crear pestañas
General_T, Riesgo_T, Aplica_T, Denuncia_T = st.tabs(["INFORMACIÓN GENERAL", "RIESGO POR PUESTOS", "RIESGO POR APLICATIVOS", "DENUNCIAS"])


with General_T:
    st.success('INFORMACIÓN GENERAL')
    if BASE_RIESGOS_UNIDAD.empty:
        st.warning("No hay datos disponibles para la selección actual.")

    else:

        Menu_conte = CONTENEDOR_M(BASE_RIESGOS_UNIDAD)
        st.markdown(Menu_conte, unsafe_allow_html=True) 

        MENU_1, MENU_2 = st.columns([2, 2])

        with MENU_1:
            MAPA_UAD = MAPA_MENU (BASE_RIESGOS_UNIDAD, BASE_MAPA)
            st.plotly_chart(MAPA_UAD, use_container_width = True)

        with MENU_2:
            PUESTOS_UAD = DONA_MENU(BASE_RIESGOS_UNIDAD,'PTO_RIESGO')
            st.plotly_chart(PUESTOS_UAD, use_container_width = True)

        PUESTOS_RIESGO = BARRAS_MENU(BASE_RIESGOS_UNIDAD)
        st.plotly_chart(PUESTOS_RIESGO, use_container_width = True)
  


with Riesgo_T:
    st.success('RIESGO POR ADMINISTRACIÓN')

    if BASE_RIESGOS_UNIDAD.empty:
        st.warning("No hay datos disponibles para la selección actual.")

    else:
        DESCO_DROP_2 = BASE_RIESGOS_UNIDAD["DESCONCENTRADA"].dropna().unique().tolist() + ['TODAS']
        DESCO_DROP = st.multiselect('Seleccionar por ADMINISTRACIÓN', 
                                     DESCO_DROP_2, 
                                     default = 'TODAS', 
                                     key = 'DESCO_DROP')
        if 'TODAS' in DESCO_DROP:
            DESCO_DROP = DESCO_DROP_2[:-1]
            

        BASE_RIESGOS_DESCO = BASE_RIESGOS_UNIDAD.query("DESCONCENTRADA == @DESCO_DROP")

        if BASE_RIESGOS_DESCO.empty:
            st.warning("NO HAY DATOS DISPONIBLES PARA LA SELECCIÓN ACTUAL.")

        else: 
            PUESTOS_T,PUESTOS_R = Puestos_Metricas(BASE_RIESGOS_DESCO)
            
            METRIC_PT, METRIC_PR = st.columns(2)
            with METRIC_PT:
                st.metric('Total de puestos', PUESTOS_T)
            with METRIC_PR:
                st.metric('Puestos de riesgo', PUESTOS_R)

            PTO_ADM_1, PTO_ADM_2 = st.columns([2,1])
            with PTO_ADM_1:
                PUESTOS_RIESGO_DESCO = BARRAS_MENU(BASE_RIESGOS_DESCO)
                st.plotly_chart(PUESTOS_RIESGO_DESCO, use_container_width = True)
            with PTO_ADM_2:
                Puesto_riesgo_conte = CONTENEDOR_PR(BASE_RIESGOS_DESCO)
                st.markdown(Puesto_riesgo_conte, unsafe_allow_html=True) 

            st.success('CONFIABILIDAD, QUEJAS & DENUNCIAS Y NIVEL DE RIESGO')
            
            QDNA_CONTE = CONTENEDOR_QDNA(BASE_RIESGOS_DESCO)
            st.markdown(QDNA_CONTE, unsafe_allow_html=True) 
            
            CON_QD_1, CON_QD_2 = st.columns([1,1])
            with CON_QD_1:
                CONFIABILIDAD_DESCO = CONFIABILIDAD (BASE_RIESGOS_DESCO)
                st.plotly_chart(CONFIABILIDAD_DESCO, use_container_width = True)
            with CON_QD_2:
                QUEJADENUN_DESCO = QUEJA_DENUNCIA (BASE_RIESGOS_DESCO)
                st.plotly_chart(QUEJADENUN_DESCO, use_container_width = True)

            NIVEL_ATE_DESCO = RIESGO_ATENCION_NIVEL(BASE_RIESGOS_DESCO)
            st.plotly_chart(NIVEL_ATE_DESCO, use_container_width = True)

            st.success('DETALLES DE APLICATIVOS POR NIVEL DE RIESGO')

            st.markdown("""<p style="text-align: justify; font-size: 15px;">
                        La tabla muestra información detallada sobre los empleados clasificados con niveles de riesgo <b>MEDIO, ALTO y CRÍTICO.</b> 
                        Incluye datos sobre los puestos que ocupan, los roles asignados, y los niveles de riesgo y atención asociados a cada 
                        puesto, basados en la desconcentrada seleccionada previamente.</p>""", unsafe_allow_html=True)


            #> Filtro NIVEL DE ATENCION
            ATENCION_FL = BASE_RIESGOS_DESCO[~BASE_RIESGOS_DESCO['NIVEL_ATENCION'].isin(['NULO', 'BAJO'])]
            DETA_APP_1 = ATENCION_FL["PUESTO_NOM"].unique().tolist()
            DETA_APP_2 = DETA_APP_1 + ['TODOS']
            DETA_APP_DROP = st.multiselect('Seleccionar por PUESTO', 
                                            DETA_APP_2, 
                                            default = 'TODOS', 
                                            key = 'DETA_APP_DROP'
                                          )
            if 'TODOS' in DETA_APP_DROP:
                DETA_APP_DROP = DETA_APP_1
            else:
                DETA_APP_DROP = [PT for PT in DETA_APP_DROP if PT in DETA_APP_1]

            DETALLES_APPS = ATENCION_FL.query("PUESTO_NOM == @DETA_APP_DROP")
            
            DETALLE, SISTEMA = APLICATIVOS_DETALLES(DETALLES_APPS)
           
            Tabla_APP = APLICATIVOS_TABLA (DETALLE)
            st.plotly_chart(Tabla_APP, use_container_width=True)

            #> Filtro EMPLEADOS
            EMPLEA_APP = SISTEMA["NOMBRE_EMP"].unique().tolist()
            EMPLEA_APP_1 = EMPLEA_APP + ['TODOS']
            
            EMPLEA_APP_DROP = st.multiselect('Seleccionar por EMPLEADO', 
                                              EMPLEA_APP_1,
                                              default = 'TODOS',
                                              key = 'EMPLEA_APP_DROP')
            
            if 'TODOS' in EMPLEA_APP_DROP:
                EMPLEA_APP_DROP = EMPLEA_APP
            else:
                EMPLEA_APP_DROP = [EMP for EMP in EMPLEA_APP_DROP if EMP in EMPLEA_APP]
            
            EMPL_SISTEM = SISTEMA.query("NOMBRE_EMP == @EMPLEA_APP_DROP")

            if EMPL_SISTEM.empty:
                st.warning("NO HAY DATOS DISPONIBLES PARA LA SELECCIÓN ACTUAL.")
            else:
                EMPLE_ROLES_CON = ROLES_EMPLEADO(EMPL_SISTEM)
                st.markdown(EMPLE_ROLES_CON, unsafe_allow_html=True) 

                ACC_APP = ACCESOS_APP (EMPL_SISTEM)
                st.plotly_chart(ACC_APP, use_container_width=True)

              


with Aplica_T:
    st.success('RIESGO POR APLICATIVOS')

    if BASE_RIESGOS_UNIDAD.empty:
        st.warning("No hay datos disponibles para la selección actual")

    else:

        APP_TOT = APLICATIVOS_TOTAL(BASE_RIESGOS_UNIDAD)
        st.plotly_chart(APP_TOT, use_container_width=True)


        CONTE_APP_R = APLICATIVOS_RIESGOS(BASE_RIESGOS_UNIDAD)
        st.markdown(CONTE_APP_R, unsafe_allow_html=True) 


        APP_PTO = APLICATIVOS_PUESTOS (BASE_RIESGOS_UNIDAD)
        st.plotly_chart(APP_PTO, use_container_width=True)

        DESCO_APP_2 = BASE_RIESGOS_UNIDAD["DESCONCENTRADA"].dropna().unique().tolist() + ['TODAS']
        DESCO_APP = st.multiselect('Seleccionar por ADMINISTRACIÓN', 
                                     DESCO_APP_2, 
                                     default = 'TODAS', 
                                     key = 'DESCO_APP')
        if 'TODAS' in DESCO_APP:
            DESCO_APP = DESCO_APP_2[:-1]
            

        BASE_RIESGOS_APP = BASE_RIESGOS_UNIDAD.query("DESCONCENTRADA == @DESCO_APP")


        if BASE_RIESGOS_APP.empty:
            st.warning("No hay datos disponibles para la selección actual")
        else:
            
            MAX_SIS_APP = APP_SISTE_MAX (BASE_RIESGOS_APP)
            st.markdown(MAX_SIS_APP, unsafe_allow_html=True) 


            MAYOR_SIS = MAYOR_SISTEMAS(BASE_RIESGOS_APP)
            st.plotly_chart(MAYOR_SIS, use_container_width=True)
            
            MAYOR_APP = APLICATIVOS_MAYOR(BASE_RIESGOS_APP)
            st.plotly_chart(MAYOR_APP, use_container_width=True)

        st.success('DETALLES DE APLICATIVOS POR UNIDAD, PUESTO Y EMPLEADO')

        st.markdown("""<p style="text-align: justify; font-size: 16px;">
                        La tabla detalla información clave sobre los empleados, sus puestos, los aplicativos asignados, 
                        el número de roles por aplicativo y el alcance de su uso. Para facilitar un análisis más profundo, 
                        los datos pueden filtrarse por <b>PUESTO</b> y <b>EMPLEADO</b> , además de estar disponibles para descarga en 
                        formato <b>Excel (.xlsx)</b>.</p>""", unsafe_allow_html=True)

        PTO_1, EMPLE_1 = st.columns([1,1])
        with PTO_1:
            PTO_UN = BASE_RIESGOS_APP["PUESTO_NOM"].unique().tolist()
            PTO_OPS = PTO_UN + ['TODOS']
            PTO_DROP = st.multiselect('Seleccionar por PUESTO', 
                                       PTO_OPS,
                                       default = 'TODOS',
                                       key = 'PTO_DROP')
            if 'TODOS' in PTO_DROP:
                PTO_DROP = PTO_OPS
            else:
                PTO_DROP = [PR for PR in PTO_DROP if PR in PTO_UN]
            
            PUESTO_FILT = BASE_RIESGOS_APP.query("PUESTO_NOM == @PTO_DROP") 

        with EMPLE_1:
            EMPL_UN = PUESTO_FILT["NOMBRE_EMP"].unique().tolist()
            EMPL_OPS = EMPL_UN + ['TODOS']
            EMPL_DROP = st.multiselect('Seleccionar por EMPLEADO', 
                                        EMPL_OPS, 
                                        default ='TODOS',
                                        key = 'EMPL_DROP')
            if 'TODOS' in EMPL_DROP:
                EMPL_DROP = EMPL_OPS
            else:
                EMPL_DROP = [ER for ER in EMPL_DROP if ER in EMPL_UN]

            EMPLE_FILT = PUESTO_FILT.query("NOMBRE_EMP == @EMPL_DROP")

        
        TAB_APPS, TAB_EXP = TABLA_APPS(EMPLE_FILT)

        st.plotly_chart(TAB_APPS, use_container_width=True)

        st.download_button(label = "DESCARGAR TABLA",
                           data = EXP_TABLA(TAB_EXP),
                           file_name = 'Detalle_Aplicativos_Empleados.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


with Denuncia_T:
    st.success('DETALLE DE LAS DENUNCIAS')

    if BASE_DENUNCIAS_UNIDAD.empty:
        st.warning("No hay datos disponibles para la selección actual")
    else:

        CONTE_DENUN = DENUNCIAS_CONTE(BASE_DENUNCIAS_UNIDAD)
        st.markdown(CONTE_DENUN, unsafe_allow_html=True) 

        ASUNTOS, DENUNCIA_QUEJA, PUESTOS = DENUNCIAS_INFO(BASE_DENUNCIAS_UNIDAD)

        VELO_INFO = VEL_DENU(DENUNCIA_QUEJA)
        st.plotly_chart(VELO_INFO, use_container_width=True)

        
        PTO_DEN, PTO_DEN_C = st.columns([2,1.5])
        with PTO_DEN:
            PUESTO_DENU = DENUNCIAS_PUESTOS(PUESTOS)
            st.plotly_chart(PUESTO_DENU, use_container_width=True)
        with PTO_DEN_C:
            DEN_PUESTO = DENUN_PTO(BASE_DENUNCIAS_UNIDAD)
            st.markdown(DEN_PUESTO, unsafe_allow_html=True) 

        CLAS_DEN, GRAP_DENC = st.columns([1.5,2])
        with CLAS_DEN:
            CLAS_DENU = CONTE_CLASS_SUNTO(BASE_DENUNCIAS_UNIDAD)
            st.markdown(CLAS_DENU, unsafe_allow_html=True)
        with GRAP_DENC:
            DENUN_CL = CLASES_DENUNCIAS (ASUNTOS)
            st.plotly_chart(DENUN_CL, use_container_width=True)
            
        DEN_EXP = TABLA_DENU(BASE_DENUNCIAS_UNIDAD)

        st.download_button(label = "DESCARGAR INFORMACION A DETALLE",
                           data = EXP_TABLA(DEN_EXP),
                           file_name = 'Detalle_Denuncias.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        



hide_streamlit_style = """
                      <style>
                      #MainMenu {visibility: hidden;}
                      footer {visibility: hidden;}
                      </style>
                      """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)