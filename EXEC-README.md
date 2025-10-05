# Proyecto: Detección de Exoplanetas con AI/ML (Kepler, K2, TESS)

## Resumen General

El proyecto propone una solución integral que combina inteligencia artificial, ciencia de datos e inclusión educativa para democratizar el acceso al conocimiento astronómico.

Desarrollamos una plataforma accesible y robusta que replica y mejora los modelos de detección de exoplanetas de la NASA, como **ExoMiner**, **ExoNet/AstroNet**, **GPC** y **Robovetter**, integrándolos en un ensamble híbrido que optimiza precisión, recall y explicabilidad.

El sistema no solo acelera la identificación científica de exoplanetas en misiones como Kepler, K2 y TESS, sino que también acerca la astronomía al público general mediante una interfaz dual: una vista científica avanzada y otra diseñada especialmente para infancias.

---

## Fundamentación e Investigaciones Previas

Este proyecto parte de investigaciones de alto impacto en la comunidad astronómica y de aprendizaje automático:

- **ExoMiner (NASA, 2021):** una red neuronal profunda que validó más de 300 nuevos exoplanetas en datos de Kepler, introduciendo el uso de ramas diagnósticas y validación cruzada estratificada.
- **ExoNet / AstroNet (Google AI, 2018–2020):** modelos convolucionales basados en vistas globales y locales de curvas de luz, pioneros en el uso de aprendizaje profundo para clasificación planetaria.
- **Robovetter (NASA Kepler Pipeline):** sistema de reglas e indicadores físicos que imitó el proceso de validación manual de astrónomos.
- **Modelos clásicos de alto rendimiento (Random Forest, LightGBM, SVM):** investigaciones recientes demostraron que, con un conjunto bien diseñado de variables de tránsito y diagnóstico, estos algoritmos alcanzan niveles de precisión comparables a redes neuronales con menor costo computacional.

El presente trabajo toma estos enfoques como base, los reproduce bajo un marco reproducible y los combina en un ensamble científico que busca lograr resultados comparables al estado del arte con mayor eficiencia y accesibilidad.

---

## Propuesta de Valor

**Problema:**  
La detección de exoplanetas aún depende de revisión manual y análisis extensos sobre millones de curvas de luz, lo que genera cuellos de botella, sesgos humanos y pérdida de oportunidades científicas.

**Solución:**  
Un modelo híbrido de IA/ML desplegado en infraestructura ligera y eficiente que:

- Automatiza la detección de exoplanetas en datos abiertos de la NASA.
- Permite la carga de datasets personalizados para validación o experimentación.
- Explica visualmente sus decisiones, acercando el proceso científico a personas sin formación técnica.

El proyecto combina rigor científico, eficiencia computacional y diseño inclusivo, permitiendo tanto el uso profesional por investigadores como la comprensión por parte de estudiantes o público infantil.

---

## Arquitectura Técnica y Pipeline Científico

El sistema está desplegado en **Microsoft Azure**, sobre una Máquina Virtual dedicada que integra todos los componentes del flujo:

### 1. Infraestructura en Azure

- **Servidor web (GoDaddy):** aloja la aplicación web accesible desde un dominio propio.
- **Backend/API REST:** orquesta peticiones entre la interfaz, la base de datos y los modelos.
- **Base de datos interna:** almacena los datasets subidos por usuarios, las predicciones y las métricas generadas por el sistema.
- **Pipeline de IA/ML:** entrenamientos, evaluaciones y despliegues de modelos, ejecutados de manera contenida dentro de la VM.

### 2. Arquitectura lógica del sistema

- **Capa de Datos:** incluye los catálogos KOI (Kepler), K2 y TOI (TESS), junto con datos cargados por los usuarios.
- **Capa de Procesamiento:** aplica detrendeo (Savitzky–Golay), correcciones de ruido, normalización y generación de vistas global/local de curvas de luz.
- **Capa de Modelado:** alberga los modelos DL (ExoMiner, ExoNet) y los clásicos (RF, LightGBM, SVM) entrenados bajo validación cruzada estratificada.
- **Capa de Ensamble:** combina resultados mediante stacking/blending calibrado, optimizando AUC-PR y recall.
- **Capa de Resultados e Interfaz:** presenta los hallazgos mediante gráficas interactivas, métricas comparativas y explicaciones visuales (importancias de variables, curvas PR/ROC).

### 3. Pipeline científico

1. Ingesta y validación de datos multi-misión.
2. Preprocesamiento con métodos reproducibles por misión (Kepler, K2, TESS).
3. Entrenamiento, evaluación y selección de modelos.
4. Ensamble final y calibración de umbrales.
5. Despliegue y visualización web de resultados.

Esta arquitectura es ligera, eficiente y resiliente, capaz de ejecutarse sin infraestructura pesada ni dependencias externas a Azure.

---

## Interfaz Web y Accesibilidad

La interfaz fue diseñada para ser accesible, intuitiva y ergonómica, enfocada en la experiencia de usuario y la comprensión sin tecnicismos.

### Vista Especializada (Investigadores)

- Permite cargar datasets personalizados, entrenar modelos, ejecutar predicciones y visualizar métricas avanzadas (AUC, PR, F1).
- Incluye un glosario interactivo de 50–100 variables que describe cada parámetro físico de las misiones NASA.

### Vista Infantil (Educativa)

- Presenta el mundo de los exoplanetas con un lenguaje sencillo y materiales visuales: videos, animaciones y ejemplos interactivos.
- El objetivo es inspirar curiosidad científica en infancias y comunidades no técnicas, utilizando diseño lúdico y colores accesibles.

La aplicación prioriza la inclusión, con un enfoque educativo que abarca diversidad cultural, de género y capacidades. Todo el sistema puede ser comprendido sin conocimiento técnico, cumpliendo los principios de accesibilidad universal.

---

## Impacto y Relevancia

- **Para la NASA y la comunidad científica:** reduce el tiempo de análisis y prioriza candidatos planetarios con mayor precisión.
- **Para la sociedad y la educación:** transforma la IA en una herramienta comprensible y motivadora, fomentando vocaciones STEM desde etapas tempranas.
- **Para el desarrollo sostenible:** contribuye a los Objetivos de Desarrollo Sostenible (ODS 4 y 9) promoviendo educación de calidad e innovación responsable.
- **Para la equidad:** diseña un espacio científico abierto, accesible e inclusivo, donde cualquier persona pueda aprender o contribuir a la búsqueda de nuevos mundos.

---

## Visión a Futuro

El proyecto busca evolucionar hacia una plataforma abierta y conectada con las misiones futuras de la NASA (JWST, PLATO, Roman), incorporando nuevas APIs y datos espectroscópicos.

En el ámbito educativo, se prevé el desarrollo de módulos interactivos y gamificados para aprendizaje astronómico inclusivo.

A largo plazo, el sistema podría escalarse a una red de ciencia ciudadana, donde cualquier usuario —científico o no— pueda analizar datos reales y participar activamente en el descubrimiento de exoplanetas.

---

## En síntesis

Una plataforma científica, educativa e inclusiva, montada en Azure, que utiliza inteligencia artificial para detectar exoplanetas en datos de la NASA y los presenta a través de una interfaz ligera, accesible y universal.

Un puente entre la ciencia avanzada y la curiosidad humana, diseñado para que cualquier persona —sin importar su edad o formación— pueda explorar el universo desde una sola página.