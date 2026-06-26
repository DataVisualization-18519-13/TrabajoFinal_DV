# Modelo dimensional para Tableau

## Decision de arquitectura

La capa `gold` se construye en Python. Tableau consume tablas ya modeladas y
solo define relaciones logicas entre ellas. No se deben hacer joins fisicos
entre todos los CSV porque se duplicarian medidas al mezclar los granos de
negocio, resena y categoria.

El modelo es una constelacion de hechos, no una estrella unica:

- `fact_business_opportunity`: una fila por negocio.
- `fact_review`: una fila por resena.
- `dim_business`: atributos descriptivos y geograficos del negocio.
- `dim_date`: calendario diario para las resenas.
- `dim_opportunity_segment`: clasificacion de calidad y popularidad.
- `dim_category`: catalogo de categorias.
- `bridge_business_category`: relacion muchos-a-muchos entre negocios y
  categorias.

## Relaciones en Tableau

Usar relaciones en la capa logica, no joins en la capa fisica.

| Tabla izquierda | Campo | Tabla derecha | Campo | Cardinalidad |
|---|---|---|---|---|
| `fact_business_opportunity` | `business_key` | `dim_business` | `business_key` | muchos-a-uno |
| `fact_business_opportunity` | `segment_key` | `dim_opportunity_segment` | `segment_key` | muchos-a-uno |
| `fact_review` | `business_key` | `dim_business` | `business_key` | muchos-a-uno |
| `fact_review` | `date_key` | `dim_date` | `date_key` | muchos-a-uno |
| `bridge_business_category` | `business_key` | `dim_business` | `business_key` | muchos-a-uno |
| `bridge_business_category` | `category_key` | `dim_category` | `category_key` | muchos-a-uno |

En Tableau 2024.2 o posterior, agregar `fact_business_opportunity`,
`fact_review` y `bridge_business_category` como tablas base. Las dimensiones
compartidas conectan los tres arboles.

Para Tableau anterior a 2024.2, usar un solo arbol logico con
`fact_business_opportunity` como tabla base:

1. Relacionar `fact_business_opportunity` con `dim_business`.
2. Desde `dim_business`, relacionar `fact_review` y
   `bridge_business_category`.
3. Relacionar `fact_review` con `dim_date`.
4. Relacionar `bridge_business_category` con `dim_category`.
5. Relacionar `fact_business_opportunity` con
   `dim_opportunity_segment`.

Este montaje conserva los granos mediante relaciones. No abrir las tablas
logicas para convertirlas en un unico join fisico.

## Medidas recomendadas

| Indicador | Definicion |
|---|---|
| Negocios | `SUM([business_count])` o `COUNTD([business_key])` |
| Resenas | `SUM([review_count])` |
| Rating de negocio | `AVG([stars])` |
| Rating de resena | `AVG([review_stars])` |
| Oportunidades | `SUM(IIF([is_opportunity], [business_count], 0))` |
| Porcentaje de oportunidad | oportunidades / negocios |

Para visuales por categoria usar `COUNTD([business_key])`. Un negocio puede
pertenecer a varias categorias; por eso la suma de negocios entre categorias
no tiene que coincidir con el total global.

## Restricciones analiticas

- `fact_business_opportunity` representa una foto del estado del negocio. No
  existe historial de `stars`, `review_count` ni `divergence_score`.
- `fact_review` permite analizar la evolucion temporal de las resenas, pero la
  fuente procesada es una muestra y solo cubre 7,129 negocios del universo
  modelado.
- Se excluyeron 183,454 resenas cuyos negocios no pasaron los filtros de
  apertura, cantidad minima de resenas o coordenadas.
- `is_first_listed` indica la primera categoria del texto de Yelp. No debe
  interpretarse como una categoria principal validada.
