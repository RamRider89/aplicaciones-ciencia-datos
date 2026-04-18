## 2. Visualización y comunicación de resultados

Con el objetivo de comunicar los hallazgos más relevantes del análisis de forma clara y accionable, se seleccionan **8 visualizaciones clave** que resumen el comportamiento de riesgo, la estructura de la cartera de clientes y los factores que más influyen en el default. Cada visualización está acompañada de una **interpretación ejecutiva** orientada al negocio.

***

### 1. Distribución global del riesgo de default en la cartera

**Visualización propuesta:**  
Gráfico de barras o donut mostrando la proporción de clientes **Default (1)** vs **No Default (0)**.

**Métrica representada:**  
Porcentaje de clientes en default y no default (`default_12m`).

**Interpretación ejecutiva:**  
La visualización confirma que **aproximadamente tres cuartas partes de la cartera presentan incumplimiento**, reflejando un **riesgo estructural elevado**. Este contexto justifica plenamente la necesidad de adoptar modelos predictivos y reglas automáticas de control de riesgo, ya que las políticas actuales permiten que el default se propague sin detección temprana.

* **Título Sugerido:** Distribución de Exposición al Riesgo en la Cartera Actual.
* **Variable/Métrica:** Predicción de Estado (Al Corriente vs. Default) en porcentaje de clientes y volumen de saldo.
* **Tipo de Gráfico:** Gráfico de Donas (Donut Chart) dual.
* **Interpretación Ejecutiva:** *"Actualmente, el 75.8% de nuestra cartera analizada presenta un riesgo de default inminente. Esta visualización demuestra que el problema no es un evento aislado, sino un riesgo sistémico. La base de clientes rentables (24.2%) es insuficiente para subsidiar las pérdidas proyectadas si no intervenimos inmediatamente."*

***

### 2. Desempeño del modelo predictivo (Curva ROC)

**Visualización propuesta:**  
Curva ROC del modelo Random Forest optimizado.

**Métrica representada:**  
Área bajo la curva ROC (AUC ≈ 0.88).

**Interpretación ejecutiva:**  
El modelo muestra una **alta capacidad de separación entre clientes sanos y clientes de alto riesgo**. Un AUC cercano a 0.90 indica que el sistema puede utilizarse con confianza para **anticipar incumplimientos antes de que se materialicen**, apoyando decisiones de congelamiento de crédito o intervención temprana.

***

### 3. Matriz de confusión: impacto operacional del modelo

**Visualización propuesta:**  
Matriz de confusión (heatmap) con valores absolutos.

**Métrica representada:**  
Verdaderos Positivos, Verdaderos Negativos, Falsos Positivos y Falsos Negativos.

**Interpretación ejecutiva:**  
La matriz permite dimensionar el impacto real del modelo en el negocio:

*   Identifica a la mayoría de los clientes que realmente caerán en default (**recall alto**).
*   Mantiene una **precisión elevada**, reduciendo el número de clientes buenos castigados injustamente.  
    Este equilibrio evidencia que el modelo **protege el capital sin destruir por completo la experiencia del cliente sano**.

***

### 4. Segmentación estratégica de clientes (K-Means)

**Visualización propuesta:**  
Gráfico de barras con el número de clientes por segmento estratégico:

*   Leales Estabilizados
*   Inactivos Bajo Límite
*   Apalancados Crónicos
*   VIP Volátiles

**Métrica representada:**  
Total de clientes por segmento.

**Interpretación ejecutiva:**  
La cartera no es homogénea. La segmentación revela que **la mayor parte del riesgo se concentra en segmentos específicos**, particularmente en clientes inactivos de bajo límite y apalancados crónicos, mientras que existe un segmento claramente más estable que debe ser protegido y fidelizado.

***

### 5. Tasa real de default por segmento de clientes

**Visualización propuesta:**  
Gráfico de barras comparando la **tasa de default real (%)** por segmento, con una línea de referencia del promedio global.

**Métrica representada:**  
Tasa de default por segmento.

**Interpretación ejecutiva:**  
Esta visualización demuestra que:

*   El segmento **Leales Estabilizados** presenta el riesgo más bajo.
*   Los segmentos **Inactivos Bajo Límite**, **Apalancados Crónicos** y **VIP Volátiles** superan claramente el promedio global.  
    El hallazgo confirma que **no todos los clientes deben recibir el mismo tratamiento**, y que la segmentación es clave para diseñar políticas diferenciadas de crédito y cobranza.

***

### 6. Importancia de variables en el riesgo de default

**Visualización propuesta:**  
Gráfico de barras horizontales con las **variables más importantes del Random Forest**.

**Métrica representada:**  
Importancia relativa de las variables (feature importance).

**Interpretación ejecutiva:**  
Las variables de mayor impacto están relacionadas con:

*   pagos atrasados,
*   porcentaje de utilización,
*   saldo promedio,
*   límite de crédito.  
    Esto confirma que el default **no depende de factores aislados**, sino de **patrones de comportamiento financiero**, reforzando la validez del enfoque predictivo y de las reglas de negocio derivadas del modelo.

***

### 7. Combinación crítica: Mora Alta vs Utilización Alta

**Visualización propuesta:**  
Gráfico de barras comparando la tasa de default según combinaciones de:

*   Mora Alta / Mora Normal
*   Utilización Alta / Utilización Normal

**Métrica representada:**  
Tasa de default (%) por combinación.

**Interpretación ejecutiva:**  
Cuando un cliente presenta **Mora Alta**, la probabilidad de default se eleva hasta niveles cercanos al **100%**, independientemente del resto de variables.  
Este hallazgo justifica de manera directa la implementación de **bloqueos automáticos de crédito** ante este comportamiento, reduciendo pérdidas críticas para la empresa.

***

### 8. Mapa visual de clientes: riesgo, segmentos y anomalías (PCA)

**Visualización propuesta:**  
Scatter plot en dos componentes PCA, coloreado por segmento, con marcadores para clientes atípicos.

**Métrica representada:**  
Distribución de clientes según comportamiento financiero, segmentación y anomalías.

**Interpretación ejecutiva:**  
El mapa visual evidencia:

*   Un núcleo de clientes estables y predecibles.
*   “Colas” de clientes con comportamientos extremos y mayor exposición al riesgo.  
    Esta visualización permite identificar **clientes de atención prioritaria** y respalda la propuesta de un enfoque **human-in-the-loop** para casos extremos.

***

## Nota sobre dashboard ejecutivo (opcional)

Para sesiones con alta dirección, estas visualizaciones pueden integrarse en un **dashboard simple en Power BI**, agrupando:

*   indicadores clave de riesgo (default global, AUC del modelo),
*   segmentación de clientes,
*   distribución de riesgo por segmento,
*   y variables críticas.

Este dashboard permitiría a los tomadores de decisiones **explorar escenarios y priorizar acciones sin depender de interpretaciones técnicas**.

***

### Cierre ejecutivo

Estas visualizaciones convierten los resultados analíticos en **insights claros, accionables y alineados al negocio**, demostrando cómo la ciencia de datos puede apoyar decisiones estratégicas para **reducir pérdidas, proteger clientes valiosos y mejorar la sostenibilidad financiera de NovaCredit Solutions**.
