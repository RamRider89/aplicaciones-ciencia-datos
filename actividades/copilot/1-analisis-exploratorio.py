#!/usr/bin/env python
# coding: utf-8

# # 📘 Introducción – Actividad 1

# En esta actividad se revisa el caso de NovaCredit Solutions quien se encuentra en una problemática con el incremento en las tasas de incumplimiento por parte de su cartera de clientes, este caso refleja el poco entendimiento del comportamiento de los clientes digitales, y para poder darlo valor a esta información, se utilizará un conjunto de datos de 10,000 clientes para descubrir las variables que realmente mueven la balanza en el riesgo crediticio, transformando la incertidumbre en probabilidad calculada.

# ## 🗽 Alcance y propósito de la Actividad 1
# 
# El objetivo de la actividad es desarrollar una solución analítica y predictiva inicial para el riesgo de default, y esto implica:
# 
# - Identificar con rigor estadístico las "señales de alerta" que preceden al incumplimiento.
# - Construir un flujo de trabajo reproducible que prepare los datos para modelos sofisticados.
# - Traducir hallazgos técnicos en insumos ejecutivos que permitan a NovaCredit rediseñar sus políticas de otorgamiento y cobranza.

# # 📖 Análisis de casos reales

# Para contextualizar el reto de NovaCredit Solutions, es fundamental observar cómo los líderes globales están utilizando la ciencia de datos en 2026 para transformar problemas complejos en ventajas competitivas. Aquí te presento dos casos de sectores estratégicos que resuenan con los pilares de tu maestría: Finanzas y Logística.

# ## 🏦 Caso JPMorgan Chase y el uso de IA Generativa

# Durante los años 2025 y 2026, JPMorgan implementó sistemas basados en LLMs especializados para revolucionar el análisis de crédito corporativo y personal, en este caso, el problema de negocio se presentó cuando los modelos de scoring tradicionales dependían exclusivamente de datos estructurados, como el historial de pagos, lo que dejaba fuera a segmentos de personas como los "nómadas digitales" o nuevas empresas con poco historial crediticio.
# 
# - Los tipos de datos utilizados fueron:
#     - Estructurados: Transacciones bancarias, saldos, límites de crédito y variables demográficas.
#     - No estructurados: Reportes anuales de empresas, noticias financieras globales, transcripciones de llamadas de servicio al cliente y comportamiento de navegación en la app bancaria.
# 
# El valor generado para la toma de decisiones fue que el sistema permitió una evaluación de riesgo dinámica, dando la posibilidad al banco de ajustar las líneas de crédito en tiempo real basándose en "señales débiles" detectadas en noticias o cambios de comportamiento. Según los reportes dados, esto ha reducido las tasas de default y ha permitido asegurar más clientes, cuando antes eran rechazados por falta de datos tradicionales.
# 
# Fuentes: 
# - [JPMorgan Press]('https://www.jpmorganchase.com/newsroom/press-releases/2025')
# - [LLM Suite named 2025 “Innovation of the Year” by American Banker]('https://www.jpmorganchase.com/about/technology/news/llmsuite-ab-award?queryID=bac22ed8ee01bc8eb84fcea0fb82bb6a')

# ## 🛻 Caso de logística y retail en DHL

# DHL utiliza un "Trend Radar" para mapear cómo las tecnologías impactarán la cadena de suministro en los próximos 10 años, situando a la IA en el centro de su estrategia. Dado al entorno global tan volátil, DHL ha transformado su logística mediante mejoras digitales alimentadas por IA.
# 
# En este escenario, a DHL se le presentaba un problema en la interrupción de rutas globales y los cambios extremos en la demanda, lo que generaba costos operativos masivos por inventario estancado o falta de stock en puntos críticos.
# 
# - Los tipos de datos utilizados durante las mejoras fueron:
#     - Estructurados (IoT): Datos de sensores de temperatura y ubicación de contenedores, niveles de inventario en almacenes.
#     - Semiestructurados/No estructurados: Reportes meteorológicos, datos de tráfico marítimo, etc...
# 
# El valor generado para la toma de decisiones fue que la ciencia de datos logro que DHL formará una anticipación estratégica, permitiendo a los ejecutivos de logística tomar decisiones sobre desvío de carga antes de que ocurrieran cuellos de botella. 
# 
# 
# Fuentes: 
# - [DHL LOGISTICS TREND RADAR 7.0]('https://www.dhl.com/global-en/delivered/global-trade/dhl-releases-logistics-trend-radar-7-0.html')
# - [Digital Twins in Logistics]('https://www.dhl.com/content/dam/dhl/global/core/documents/pdf/glo-core-digital-twins-in-logistics.pdf')

# ## 🪶 Reflexión de los casos estudiados

# Estos dos casos presentados demuestran que el éxito del negocio tiene mucho que ver con el uso que se le da a los datos y las diferentes herramientas para la extracción y aprovechamiento de los mismos.
# 
# Por ejemplo, en el area de finanzas, con JPMorgan, el valor se demostró en la transparencia y el contexto del dato, pudiendo así explicar por qué el riesgo aumentó. Por otro lado, en el area de logística, con DHL, el valor de la información se demostró en la operatividad y movimiento de recursos físicos.
# 
# Para NovaCredit, el reto será integrar los datos estructurados del dataset con una metodología que permita entender las variables que mayor influyen en el comportamiento de los clientes y de que manera se pueden intervenir mediante estrategias de cobranza preventiva.

# # 💻 Entorno Python para la Actividad 1

# Se creó un nuevo entorno de trabajo optimizada para el análisis de datos masivos y la automatización de procesos de calidad seleccionado `Python 3.10` y herramientas de `Scikit-Learn` para Pipelines y Transformaciones.
# 
# A continuación, describo los componentes clave del entorno funcional:
# 
# Especificaciones del Entorno para el análisis de datos:
# - Núcleo de ejecución sobre Python 3.10, que es una versión "LTS" que es el estándar actual para ciencia de datos.
# - Gestión híbrida de paquetes mediante `Conda` y `Pip`, para instalar binarios pesados mediante **conda** como `sklearn` y **pip** para librerías de manipulación como `pandas`, se aseguran que las dependencias compartidas se resuelvan sin conflictos de versiones.
# 
# Librerías principales:
# 1. **Pandas**: Como motor principal del entorno.
# 2. **Scikit-Learn**: para Pipelines y Transformaciones.

# ## 🏠 Selección del conjunto de datos

# Para cumplir con los requisitos de la Actividad 1 y asegurar todos los pasos requeridos durante su desarrollo, se utilizará el conjunto de datos de **novacredit_clientes.csv** provisto en Canvas de Tecmilenio.

# ### 📒 Acerca del conjunto de datos

# El archivo `novacredit_clientes.csv` está compuesto por 10,000 registros y 16 variables que integran dimensiones demográficas, financieras y conductuales de los clientes de NovaCredit Solutions, este archivo constituye la base analítica para identificar patrones de incumplimiento y segmentar el riesgo crediticio.
# 
# ### Composición del Dataset
# El conjunto de información se puede categorizar en cuatro pilares fundamentales:
# 
# - **Perfil demográfico y de relación**, describiendo la edad del cliente, la región geográfica, el canal por el cual fue adquirido y su antigüedad en meses con la institución.
# - **Capacidad y uso financiero**: Define el ingreso mensual, el límite de crédito asignado, el saldo promedio mantenido y el porcentaje de utilización de los recursos disponibles.
# - **Indicadores de comportamiento y riesgo**: Registra variables críticas como el número de transacciones mensuales, pagos atrasados en el último año y el volumen de reclamos generados por el usuario.
# - **Definición contractual y objetivo**: Detalla el tipo de producto como tarjeta, préstamo o línea de crédito, la modalidad del contrato, el método de pago y la variable objetivo `default_12m`, que señala si el cliente incurrió en incumplimiento.

# ### 🕵️‍♂️ Tipos de datos

# | Variable                 | Tipo de dato        | Descripción                                                                                                                                           |
# | ------------------------ | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
# | edad                     | Numérica (entero)   | Edad del cliente en años, con valores entre 18 y 75.                                                                                                  |
# | region                   | Categórica          | Región geográfica del cliente (Norte, Centro, Sur).                                                                                                   |
# | canal_adquisicion        | Categórica          | Canal de adquisición del cliente (Web, App, Sucursal, Referido).                                                                                      |
# | ingreso_mensual          | Numérica (continua) | Ingreso mensual estimado del cliente; puede contener valores nulos, especialmente en el segmento Básico.                                              |
# | antiguedad_cliente_meses | Numérica (entero)   | Tiempo, en meses, que el cliente ha permanecido activo en la empresa.                                                                                 |
# | segmento_cliente         | Categórica          | Segmento del cliente según su perfil (Básico, Plus, Premium).                                                                                         |
# | limite_credito           | Numérica (continua) | Límite de crédito asignado al cliente, asociado a su perfil financiero.                                                                               |
# | saldo_promedio           | Numérica (continua) | Saldo promedio utilizado durante el periodo de análisis; presenta valores atípicos en clientes de alto valor o riesgo.                                |
# | porcentaje_utilizacion   | Numérica (continua) | Proporción del crédito utilizado respecto al límite asignado, con valores entre 0 y 1.                                                                |
# | num_transacciones_mes    | Numérica (entero)   | Número promedio de transacciones mensuales realizadas por el cliente.                                                                                 |
# | pagos_atrasados_12m      | Numérica (entero)   | Número de pagos registrados con atraso en los últimos 12 meses.                                                                                       |
# | reclamos_12m             | Numérica (entero)   | Número de reclamos o incidencias reportadas en los últimos 12 meses.                                                                                  |
# | tipo_producto            | Categórica          | Tipo de producto financiero contratado (Tarjeta, Préstamo, Línea de crédito).                                                                         |
# | tipo_contrato            | Categórica          | Modalidad contractual del cliente (Mensual, Anual).                                                                                                   |
# | metodo_pago              | Categórica          | Método de pago principal del cliente (Transferencia, Débito automático, Efectivo); puede contener valores nulos.                                      |
# | default_12m              | Binaria (0/1)       | Variable objetivo que indica si el cliente incurrió en un incumplimiento significativo de pago en los últimos 12 meses (1 = default, 0 = no default). |

# ### 🔗 Carga el conjunto de datos en url pública

# Se utiliza la url pública del Dataset previamente publicado en GitHub: [_URL](https://raw.githubusercontent.com/RamRider89/aplicaciones-ciencia-datos/refs/heads/main/actividades/actividad-1/novacredit_clientes.csv)

# # 💻 Desarrollo de la Actividad 1

# **A continuación:**
# 
# 🧰 Se importan las librerías necesarias para el análisis y la visualización como Pandas Profiling, PyJanitor, entre otras.
# 
# 🔧 Se carga el dataset y se muestra una tabla descriptiva del dataset.

# ## 🧰 Carga de todas las librerías necesarias

# In[1]:


# ------------------------------------------------------------------------------
# LIBRERIAS
# ------------------------------------------------------------------------------
import sys
import os
# warnings
import warnings
warnings.filterwarnings('ignore')

# Prueba rápida de carga
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tabulate import tabulate

    # Herramientas de Scikit-Learn para Pipelines y Transformaciones
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

    print(f"Python: {sys.version.split()[0]}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print("-" * 80)
    print("🚀 ¡Entorno preparado!")
    print("-" * 80)

except Exception as e:
    print(f"❌ Error de entorno, favor de instalar las dependencias necesarias: {e}")

warnings.filterwarnings('ignore')


# ## 🗂 Variables de entorno

# In[2]:


# ------------------------------------------------------------------------------
# VARIABLES DE ENTORNO
# ------------------------------------------------------------------------------
# colores tecmilenio
_color_tecmi_light="#26d07c"
_color_primary="#007bff"
_color_gray="#6c757d"
_color_white="#ffffff"
_color_danger="#ff0000"
_color_success="#28a745"
_color_info="#17a2b8"
_color_warning="#ffc107"

# configuración de estilo visual
sns.set_theme(style="whitegrid")


# ### 🗂 Funciones auxiliares

# In[3]:


# --------------------------------------------------------------------------
# 🤖 display mensaje
# --------------------------------------------------------------------------
def display_mensaje(msj):
    print("\n" + "="*80)
    print("\033[1m --- " + msj + " --- \033[0m")
    print("="*80)


# In[4]:


# ------------------------------------------------------------------------------
# 🤖 diccionario builder
# ------------------------------------------------------------------------------

def dictionary_builder(columnas_categoricas, msj):

    _dict_unique_values_ = {}

    # recorremos las columnas categoricas para obtener sus valores
    display_mensaje(msj)
    # recorreos las columnas categoricas
    for col in columnas_categoricas.columns:
        # obtenemos los vals unicos
        unique_values = columnas_categoricas[col].unique()
        # asignamos los valores unicos al dict
        _dict_unique_values_[col] = columnas_categoricas[col].unique()


    # para mostrar el resultado del diccionario en pantalla
    # convertimos el diccionario a un conjunto de listas
    data = [[key] + list(value) for key, value in _dict_unique_values_.items()]

    # definimos los titulos de la tabla
    headers = ["Columna"] + [f"Valor único {i+1}" for i in range(len(max(_dict_unique_values_.values(), key=len)))]
    # imprimiendo
    print(tabulate(data, headers=headers, tablefmt="grid"))

    return _dict_unique_values_;


# # 💾 Carga de los datos

# Para iniciar el análisis se utiliza la url pública del dataset en **Canvas**.

# In[5]:


# ------------------------------------------------------------------------------
# CARGA DEL DATASET
# ------------------------------------------------------------------------------
# URL pública del dataset
_URL_ = 'https://raw.githubusercontent.com/RamRider89/aplicaciones-ciencia-datos/refs/heads/main/actividades/actividad-1/novacredit_clientes.csv'
#_URL_ = "./product_sales_dataset_final.csv"
# config
pd.set_option('display.max_columns', None)

# leyendo el archivo csv mediante pandas
df_nova = pd.read_csv(_URL_)


# # 🔍 Análisis exploratorio inicial

# A continuación se inicia con el análisis inicial del dataset.

# ## 🗂 Descripción del dataframe

# In[6]:


display_mensaje("Estructura inicial del DataFrame")
df_nova.head()


# ## 🗂 Información de la estructura

# In[7]:


df_nova.info()


# In[8]:


display_mensaje("Estructura del Dataset")
print(f"Dataset shape: {df_nova.shape}")
print(f"Número de filas: {df_nova.shape[0]}")
print(f"Número de columnas: {df_nova.shape[1]}")


# ## 🗂 Análisis de columnas por tipo

# Definimos los tipos de columnas para facilitar los siguientes procesos

# In[9]:


# grupos de columnas
_COLUMNAS_NUMERICAS_ = ['edad', 'ingreso_mensual', 'antiguedad_cliente_meses', 'limite_credito', 
                  'saldo_promedio', 'porcentaje_utilizacion', 'num_transacciones_mes', 
                  'pagos_atrasados_12m', 'reclamos_12m']

_COLUMNAS_CATEGORICAS_ = ['region', 'canal_adquisicion', 'tipo_producto', 'tipo_contrato', 'metodo_pago']
_COLUMNAS_ORDINALES_ = ['segmento_cliente']


# In[10]:


# ------------------------------------------------------------------------------
# valores unicos en cols categóricas
# ------------------------------------------------------------------------------
# en base a nuestras columnas categoricas: _Sex_, _Smoker_, _Region_
columnas_categoricas = df_nova[_COLUMNAS_CATEGORICAS_]
columnas_ordinales = df_nova[_COLUMNAS_ORDINALES_]

# definimos el diccionario de valores unicos
_DICT_CAT_ = dictionary_builder(columnas_categoricas, "Variables categóricas y valores únicos")
_DICT_ORDINAL_ = dictionary_builder(columnas_ordinales, "Variables ordinales y valores únicos")


# ## 🗂 Estadísticas básicas

# In[11]:


df_nova[_COLUMNAS_NUMERICAS_].describe()


# ## 🔍 Revisión a profundidad

# ### 📊 Distribución de variable objetivo default_12m

# In[12]:


# ------------------------------------------------------------------------------
# Distribución de variable objetivo default_12m
# ------------------------------------------------------------------------------
display_mensaje("Distribución de variable default_12m")
print("Porcentaje de distribución:")
dist_default = df_nova['default_12m'].value_counts(normalize=True) * 100
print(dist_default)

plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df_nova, 
    x='default_12m', 
    palette=[_color_gray, _color_primary])

plt.title('Distribución de riesgo de default')
plt.xlabel('0: No Default, 1: Default')
plt.ylabel('')
plt.show()


# Se puede apreciar como el 75.84% de la cartera de clientes está en condición de default, lo que indica un posible fallo en las políticas de otorgamiento de crédito pasadas. El historial de clientes en el dataset refleja cómo es un cliente moroso, pero incluye poca información para saber cómo es un "buen cliente".

# ### 📊 Detección de nulos

# In[13]:


# ------------------------------------------------------------------------------
# Detección de nulos
# ------------------------------------------------------------------------------
display_mensaje("Detección de Valores Nulos")
nulos = df_nova.isnull().sum()
nulos_vars = nulos[nulos > 0]
print(nulos_vars)

if not nulos_vars.empty:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=nulos_vars.index, y=nulos_vars.values, palette='Reds_r')
    plt.title('Variables con valores nulos')
    plt.ylabel('')
    plt.xlabel('')
    plt.show()


# El gráfico de nulos confirma que faltan 744 datos de ingresos y 206 en método de pago.

# ### 🕵️‍♂️ Valores únicos en variables categóricas

# In[14]:


# ------------------------------------------------------------------------------
# Valores únicos en variables categóricas
# ------------------------------------------------------------------------------
display_mensaje("Valores unicos en variables categóricas")
cat_cols = df_nova.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\nColumna '{col}':")
    print(df_nova[col].unique())


# Al imprimir los valores únicos podemos detectar los errores de codificación en los valores como 'BÃ¡sico', 'PrÃ©stamo', etc. Estos errores deberán ser tratados en las siguientes fases.

# ### 🕵️‍♂️  Detección de outliers en variables financieras

# In[15]:


# ------------------------------------------------------------------------------
# Detección de outliers en variables financieras
# ------------------------------------------------------------------------------
display_mensaje("Resumen estadístico de variables financieras")
fin_cols = ['ingreso_mensual', 'limite_credito', 'saldo_promedio']
print(df_nova[fin_cols].describe())

plt.figure(figsize=(12, 5))
for i, col in enumerate(fin_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df_nova[col], color='skyblue')
    plt.title(f'Distribución y Outliers:\n{col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()


# Al revisar los Boxplots de las variables financieras, se observa una fuerte asimetría hacia la derecha, por ejemplo, el `ingreso promedio` es de $16,000, pero hay valores máximos de hasta $196,000. Mientras que el `límite de crédito` salta de un promedio de $46,400 a un máximo extremo de $781,000.
# 
# Si esto no se escala o estandariza los modelos podrán malinterpretar la información en los extremos.

# ## 💽 Resumen del dataset

# 
# De acuerdo con el analisís del conjunto de datos `novacredit_clientes.csv`, se presenta un resumen del diagnóstico inicial que servirá de base para las siguientes fases del proyecto:
# 
# ### Diagnóstico del Dataset
# El conjunto de datos cuenta con 10,000 registros, y se identifican los siguientes puntos clave:
# 
# - La variable objetivo `default_12m` presenta una distribución muy cargada de clientes con incumplimiento, donde el 75.84% de los registros corresponden a casos de incumplimiento y el 24.16% a no incumplimiento.
# 
# - Calidad de datos y valores nulos:
# 
#     - La variable `ingreso_mensual` tiene 744 valores faltantes. Según el diccionario.
#     - Mientras que la variable `metodo_pago` tiene 206 valores faltantes.
# 
# - Se observan caracteres especiales en variables categóricas, como por ejemplo se puede observar "BÃ¡sico" en lugar de "Básico", estas variables categorizas se deben normalizarse para asegurar la consistencia de la información.
# 
# - En cuanto a los datos financieros se puede apreciar que el límite de crédito promedio es de $56,506, pero existen valores extremos de hasta $781,006, lo que sugiere la presencia de clientes de alto valor que requieren un análisis de outliers.

# # ⚙️ Preparación de datos en Python

# A continuación se inicia la fase de limpieza y transformación de datos.

# ## 📖 Tratamiento y normalización de textos

# En el análisis exploratorio previo se detectaron problemas de codificación como por ejemplo en los textos: "BÃ¡sico", "DÃ©bito automÃ¡tico", entre otros.
# 
# Estas anomalías textuales serán tratadas antes de que entren al flujo de codificación categórica para asegurar la integridad de las categorías.

# In[16]:


# ------------------------------------------------------------------------------
# Corrección de codificación en cadenas de texto 
# ------------------------------------------------------------------------------
def limpiar_texto_categorico(df):
    """
    Esta es una funcion para tratar anomalías textuales en variables categoricas.
    El objetivo es corregir los errores de codificacion
    """
    df_clean = df.copy()
    diccionario_correccion = {
        'Ã¡': 'á', 
        'Ã©': 'é', 
        'Ã­': 'í'
    }
    
    # limpieza solo a columnas categoricas
    cols_categoricas = df_clean.select_dtypes(include=['object']).columns
    for col in cols_categoricas:
        df_clean[col] = df_clean[col].replace(diccionario_correccion, regex=True)
        
    return df_clean

# recodificacion de caracteres
df_nova_limpio = limpiar_texto_categorico(df_nova)
display_mensaje("Recodificación finalizada")
df_nova_limpio.head()


# De esta manera se han eliminado los errores de codificación en el dataset.

# # 📑 Manejo de valores nulos

# De acuerdo con el analysis previo, se tiene la presencia de valores nulos tanto en `ingreso_mensual` como en el `metodo_pago`.
# 
# Dado que no es posible imputar el ingreso con el valor de la mediana global, ya que pueden existir diferente tipos de ingresos, por ejemplo un cliente "Premium" es distinto a un cliente "Básico", se realizará una imputación agrupada de acuerdo al tipo de cliente.

# In[17]:


# ------------------------------------------------------------------------------
# Imputación de valores nulos basada en reglas de negocio
# ------------------------------------------------------------------------------

# imputando el ingreso_mensual usando la mediana de su segmento_cliente
mediana_ingreso_por_segmento = df_nova_limpio.groupby('segmento_cliente')['ingreso_mensual'].transform('median')
df_nova_limpio['ingreso_mensual'] = df_nova_limpio['ingreso_mensual'].fillna(mediana_ingreso_por_segmento)

# imputando el metodo_pago con una categoría de moda
moda_pago = df_nova_limpio['metodo_pago'].mode()[0]
df_nova_limpio['metodo_pago'] = df_nova_limpio['metodo_pago'].fillna(moda_pago)

display_mensaje("Imputación finalizada")
print("Valores nulos restantes:")
print("-" * 30)
print(df_nova_limpio.isnull().sum())


# Después de la imputación realizada ya no se cuentan con valores nulos en el dataset.

# ## 🔍 Tratamiento de outliers

# En el análisis previo se pudieron observar que ciertas variables como `limite_credito` presentan clientes con valores muy extremos, muy fuera del promedio. Para evitar que estos valores extremos sesguen futuros modelos, se aplicará una técnica para limitar los valores extremos a un percentil alto, en lugar de eliminarlos, ya que un límite alto en un cliente "Premium" no es un error, es información valiosa.

# In[18]:


# ------------------------------------------------------------------------------
# Tratamiento de outliers 
# ------------------------------------------------------------------------------

def set_limite_percentil_alto(df, columna, limite_superior_percentil=0.99):
    """
    El script limita los valores extremos al percentil 99 para no perder 
    registros valiosos de clientes Premium.
    """
    limite = df[columna].quantile(limite_superior_percentil)
    df[columna] = np.where(df[columna] > limite, limite, df[columna])
    return df

# aplicando limites a las variables financieras
df_nova_limpio = set_limite_percentil_alto(df_nova_limpio, 'limite_credito')
df_nova_limpio = set_limite_percentil_alto(df_nova_limpio, 'saldo_promedio')

display_mensaje("Limites de datos financieros ajustados")


# Mediante el ajuste de valores extremos sin el riesgo de perder información, se entregará in dataset de mayor calidad para los siguientes modelos.

# ## 🤖 Pipeline de transformación, codificación y automatización

# Para garantizar que el código pueda ser reutilizable y preparar el conjunto de datos para su ingreso a los modelos predictivos, se ha generado un pipeline donde se incluyen las transformaciones finales mediante Scikit-Learn.
# 
# - Las variables ordinales se transformarán mediante `OrdinalEncoder` ya que estas tienen un orden lógico, como por ejemplo (Básico < Plus < Premium).
# - Las variables nominales no tienen jerarquía, ya que solo representan ciertas zonas o categorías, por lo cual se utilizará OneHotEncoder para estas variables.
# - Mientras que las variables numéricas serán escaladas mediante StandardScaler para normalizar su peso en el modelo.

# In[19]:


# ------------------------------------------------------------------------------
# Pipeline de Transformación Final
# ------------------------------------------------------------------------------

# separando características y variable objetivo
X = df_nova_limpio.drop(columns=['default_12m']) # caracteristicas
y = df_nova_limpio['default_12m'] # objetivo en y

# grupos de columnas
cols_numericas = ['edad', 'ingreso_mensual', 'antiguedad_cliente_meses', 'limite_credito', 
                  'saldo_promedio', 'porcentaje_utilizacion', 'num_transacciones_mes', 
                  'pagos_atrasados_12m', 'reclamos_12m']

cols_nominales = ['region', 'canal_adquisicion', 'tipo_producto', 'tipo_contrato', 'metodo_pago']
cols_ordinales = ['segmento_cliente']

# transformaciones para cada grupo
transformador_numerico = Pipeline(steps=[
    ('scaler', StandardScaler())
])

transformador_nominal = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# transformacion ordinal en orden de categoria
orden_segmentos = [['Básico', 'Plus', 'Premium']]
transformador_ordinal = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=orden_segmentos))
])

# procesamiento final
preprocesador = ColumnTransformer(
    transformers=[
        ('num', transformador_numerico, cols_numericas),
        ('nom', transformador_nominal, cols_nominales),
        ('ord', transformador_ordinal, cols_ordinales)
    ])

# preprocesamiento en todo el conjunto de características
X_preprocesado = preprocesador.fit_transform(X)

# aplicando los nombres de columnas originales + codificacion
nombres_nominales = preprocesador.named_transformers_['nom'].named_steps['onehot'].get_feature_names_out(cols_nominales)
nombres_columnas_finales = cols_numericas + list(nombres_nominales) + cols_ordinales

# Dataset final
df_final_preparado = pd.DataFrame(X_preprocesado, columns=nombres_columnas_finales)
df_final_preparado['default_12m'] = y.values

display_mensaje("Vista del Dataset Final Preparado")
df_final_preparado.head()

df_final_preparado.to_csv("df_final_preparado.csv", index=False)


# Con este flujo, se ha transformado el conjunto de datos con errores en un dataset limpio y codificado, listo para ser ingresado a los modelos de machine learning.

# ## 📖 Verificación del dataset mejorado

# ### 📊 Detección de nulos

# In[20]:


# ------------------------------------------------------------------------------
# Detección de nulos
# ------------------------------------------------------------------------------
display_mensaje("Detección de Valores Nulos")
nulos = df_final_preparado.isnull().sum()
nulos_vars = nulos[nulos > 0]
print(nulos_vars)

if not nulos_vars.empty:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=nulos_vars.index, y=nulos_vars.values, palette='Reds_r')
    plt.title('Variables con valores nulos')
    plt.ylabel('')
    plt.xlabel('')
    plt.show()
else:
    display_mensaje("No existen valores nulos en el dataset")


# ### 🕵️‍♂️  Detección de outliers en variables financieras

# In[21]:


# ------------------------------------------------------------------------------
# Detección de outliers en variables financieras
# ------------------------------------------------------------------------------
display_mensaje("Resumen estadístico de variables financieras")
fin_cols = ['ingreso_mensual', 'limite_credito', 'saldo_promedio']
print(df_final_preparado[fin_cols].describe())

plt.figure(figsize=(12, 5))
for i, col in enumerate(fin_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df_final_preparado[col], color='skyblue')
    plt.title(f'Distribución y Outliers:\n{col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()


# El ajuste de valores extremos es notorio en el nuevo dataset.

# # 📖 Resumen del dataset mejorado

# El conjunto de datos original se ha transformado de un estado sucio y propenso a errores a un dataset mejorado, estandarizado y listo para ser incluido en los siguientes modelos: 
# Después de las mejoras aplicadas se logro lo siguiente:
# 
# - Se eliminaron los valores nulos en las variables, por ejemplo el `ingreso_mensual` ahora refleja una estimación realista al haber sido imputado según la mediana de cada segmento de cliente, preservando la lógica económica, mientras que el `metodo_pago` fue completado de forma consistente.
# 
# - Las fallas de codificación de texto fueron corregidas, y enseguida, estas categorías se transformaron en vectores binarios mediante la codificaciones *OneHot Encoding*. Por otro lado la variable `segmento_cliente` ahora posee una jerarquía matemática clara mediante la codificación con Ordinal Encoding.
# 
# - El manejo de outliers se realizó con el ajuste de los valores máximos en las variables financieras garantizando que los clientes *Premium* no distorsionen los modelos basados en distancias, y sin la necesidad de eliminar sus registros de la base de datos.
# 
# - Todas las variables continuas han sido transformadas mediante StandardScaler, por lo que ahora tienen el mismo peso inicial frente al modelo, previniendo sesgos algorítmicos.
# 
# - Finalmente se ha generado un Pipeline inicial de las reglas de transformación mediante Scikit-Learn.

# # 🧾 Reflexión final

# Dado el actual escenario para **NovaCredit Solutions**, donde la tasa de default alcanza un alarmante 75.84%, la calidad de los datos definirá la mejor estrategia de mitigación en los siguientes modelos. 
# HHablando específicamente de los datos y los errores encontrados en el dataset original, es importante resaltar que si se hubiera introducido el dataset original directamente a un modelo de Machine Learning, los resultados habrían sido estadísticamente incorrectos. Por ejemplo, los errores tipográficos habrían creado dimensiones inexistentes, esto causa de los errores de codificación. Asimismo, imputar los ingresos faltantes con una media global habría alterado la capacidad de pago del segmento más bajo, llevando al algoritmo a subestimar su riesgo real.
# 
# Debemos recordar que un modelo entrenado con datos de mala calidad genera predicciones incorrectas, y por ejemplo, si el modelo clasificará erróneamente a un buen cliente como "alto riesgo", NovaCredit podría perder rentabilidad y retención de buenos clientes. Por otro lado, si clasificará a un cliente moroso como "bajo riesgo", la empresa podría absorber el impacto directo en su flujo de caja.
# 
# Esta primer practica afianza el entendimiento de que un dataset limpio y preparado garantiza que las decisiones sean explicables y justas, evitando sesgos que podrían discriminar injustamente a ciertos sectores demográficos por errores en la recolección de datos.
