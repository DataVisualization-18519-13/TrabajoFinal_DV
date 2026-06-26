# Inteligencia de Mercado Local — Análisis de Oportunidades de Negocio en Yelp

## Información del curso

| Campo | Detalle |
|---|---|
| Curso | Data Visualization |
| Carrera | Ciencias de la Computación — UPC |
| Profesor | Carlos Adrian Alarcon Delgado |
| Entrega actual | E3 — Modelado Analítico y Métricas Derivadas |

## Integrantes

| Código | Nombre |
|---|---|
| U202215375 | Ricardo Rafael Rivas Carrillo |
| U202216148 | Salvador Diaz Aguirre |
| U202212243 | Joaquin Fernando Arévalo Alcántara |

---

## Pregunta analítica central

> ¿Qué combinación de categoría, ubicación, atributos y evolución temporal caracteriza los segmentos de mayor oportunidad para abrir o expandir un negocio local en las ciudades cubiertas por el Yelp Open Dataset?

---

## Dataset

| Campo | Detalle |
|---|---|
| Nombre | Yelp Open Dataset (Academic) |
| Fuente | Yelp Inc. |
| Registros (negocios) | 150,346 originales → 119,698 activos |
| Registros (reseñas) | 775,955 (muestra procesada) |
| Período | 2005 – 2022 |
| Cobertura | 11 áreas metropolitanas EE.UU. y Canadá |
| Licencia | Yelp Dataset Challenge License |
| Descarga | https://business.yelp.com/data/resources/open-dataset/ |

> Los archivos raw (`data/raw/`) no se versionan en Git por tamaño. Descarga manual requerida.

---

## Stack tecnológico

| Capa | Herramienta |
|---|---|
| Procesamiento | Python 3.11, pandas, numpy |
| Pipeline | Patrón medallion (bronze → silver → gold) |
| Storage | Google Cloud Storage (`gs://yelp-dv-grupo/`) |
| Visualización | Tableau Desktop / Tableau Public |
| Notebooks | Jupyter |
| Componente avanzado (E6) | scikit-learn (PCA / t-SNE) |

---

## Estructura del repositorio

```
TrabajoFinal_DV/
  src/yelp_analysis/
    config.py              ← configuración tipada con dataclasses
    storage.py             ← capa de abstracción local / GCP
    medallion/
      bronze.py            ← carga y normalización del dato crudo
      silver.py            ← features analíticas y métricas derivadas
      gold.py              ← modelo dimensional para Tableau
    pipeline/
      one_shot.py          ← orquestador: corre bronze → silver → gold
  configs/
    pipeline.yaml          ← parámetros del pipeline (filtros, rutas, modo GCP)
  notebooks/
    02_perfilado_limpieza.ipynb   ← E2: perfilado, diccionario, limpieza
    03_modelado_metricas.ipynb    ← E3: modelado, comparativa, validación
  scripts/
    eda_dashboard.py       ← script EDA original (referencia histórica)
    add_geo_to_businesses.py
  data/
    processed/             ← parquets de reviews preprocesados
  outputs/                 ← generado por el pipeline (no versionado)
  docs/                    ← informes de entrega
  credentials/             ← credenciales GCP (no versionado, ver instrucciones)
  .env.example             ← variables de entorno requeridas
  pyproject.toml           ← paquete instalable
  README.md
```

---

## Métricas derivadas clave

| Métrica | Fórmula | Uso |
|---|---|---|
| `log_review_count` | `log1p(review_count)` | Normaliza distribución sesgada de demanda |
| `stars_norm` | `(stars - min) / (max - min)` | Calidad normalizada [0,1] |
| `log_rc_norm` | `(log_rc - min) / (max - min)` | Popularidad normalizada [0,1] |
| `divergence_score` | `stars_norm - log_rc_norm` | Identifica joyas ocultas (alta calidad, baja visibilidad) |
| `quadrant` | umbral 0.5 en ambos ejes | Segmento de oportunidad del negocio |
| `first_listed_category` | primera etiqueta de `categories` | Orden original; no implica categoria principal |

---

## Cómo reproducir el pipeline

### Requisitos

- Python 3.11+
- Google Cloud SDK instalado y autenticado
- Acceso al proyecto GCP `yelp-dv-grupo` (solicitar a Joaquin)

### Instalación

```bash
git clone https://github.com/DataVisualization-18519-13/TrabajoFinal_DV.git
cd TrabajoFinal_DV
pip install -e .
```

### Credenciales GCP

```bash
mkdir credentials
gcloud auth login
gcloud config set project yelp-dv-grupo
gcloud iam service-accounts keys create credentials/gcp-key.json \
  --iam-account=yelp-dv-sa@yelp-dv-grupo.iam.gserviceaccount.com
```

### Variables de entorno

Crea un archivo `.env` basado en `.env.example`:

```
GOOGLE_APPLICATION_CREDENTIALS=credentials/gcp-key.json
GCP_PROJECT=yelp-dv-grupo
GCS_BUCKET=yelp-dv-grupo
```

### Correr el pipeline

**Modo GCP (recomendado):**

```bash
# PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="credentials/gcp-key.json"
python src/yelp_analysis/pipeline/one_shot.py
```

**Modo local:**

Cambia en `configs/pipeline.yaml`:

```yaml
run:
  storage_mode: local
data:
  raw_dir: TU_RUTA_LOCAL/Yelp JSON
```

Luego:

```bash
python src/yelp_analysis/pipeline/one_shot.py
```

### Output esperado

```
=== PIPELINE YELP DV — ONE SHOT ===
── Bronze ──
[bronze] business: 119,698 filas → gs://yelp-dv-grupo/bronze/
[bronze] reviews:  775,955 filas → gs://yelp-dv-grupo/bronze/
── Silver ──
[silver] business:        119,698 filas → gs://yelp-dv-grupo/silver/
[silver] reviews:         775,955 filas → gs://yelp-dv-grupo/silver/
[silver] categories:          687 filas → gs://yelp-dv-grupo/silver/
[silver] monthly_reviews:     202 filas → gs://yelp-dv-grupo/silver/
── Gold ──
[gold] dim_business
[gold] dim_date
[gold] dim_category
[gold] dim_opportunity_segment
[gold] bridge_business_category
[gold] fact_business_opportunity
[gold] fact_review
=== PIPELINE COMPLETO ===
```

La configuracion de relaciones, cardinalidades y medidas de Tableau se
documenta en `docs/tableau_star_schema.md`.

---

## Estado de entregas

| Entrega | Semana | Estado | Artefacto principal |
|---|---|---|---|
| E1 — Propuesta | 3 | ✅ Entregado | `docs/PC1_DataVisualization.pdf` |
| E2 — Perfilado y limpieza | 5 | ✅ Entregado | `notebooks/02_perfilado_limpieza.ipynb` |
| E3 — Modelado y métricas | 7 | ✅ Entregado | `notebooks/03_modelado_metricas.ipynb` |
| E4 — Segmentación y fuentes Tableau | 11 | 🔲 Pendiente | — |
| E5 — Dashboard alpha | 13 | 🔲 Pendiente | — |
| E6 — Dashboard final + PCA/t-SNE | 15 | 🔲 Pendiente | — |

---

## Repositorio GCP

| Recurso | Detalle |
|---|---|
| Proyecto | `yelp-dv-grupo` |
| Bucket | `gs://yelp-dv-grupo/` |
| Capas | `raw/`, `processed/`, `bronze/`, `silver/`, `gold/` |
| Service Account | `yelp-dv-sa@yelp-dv-grupo.iam.gserviceaccount.com` |
