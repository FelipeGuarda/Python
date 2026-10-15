# Nuestro protocolo de calidad de datos, leído contra Silva-Rodríguez et al. (2025)

**Fundación Mar Adentro · Bosque Pehuén**
Documento interno. Edición 2026-10-15.

> Silva-Rodríguez, E. A., Cortés, E. I., Vasquez-Ibarra, V., Gálvez, N., Cusack, J.,
> Ohrens, O., Moreira-Arce, D., Farías, A. A. & Infante-Varela, J. (2025).
> *A protocol for error prevention and quality control in camera trap datasets.*
> Journal of Applied Ecology. DOI: 10.1111/1365-2664.70010

Este documento explica, al equipo y a quien coordina el trabajo, qué hace nuestra cadena de
datos de cámaras trampa para que sus resultados se puedan creer, usando como vara de medida un
protocolo publicado de control de calidad en fototrampeo. El detalle operativo vive en
`MANUAL-SALUD-DATOS.md`; aquí está el argumento.

---

## 1. Resumen

**Lo que propone el paper.** Silva-Rodríguez et al. parten de una pregunta: *¿puede un lector
confiar en una base de datos de fototrampeo?* Revisaron 147 artículos publicados entre 2021 y
2023 y encontraron que casi ninguno da los elementos para responderla: el 4,8 % reporta alguna
métrica de control de calidad y el 0,7 % dice cómo corrigió errores de fecha y hora. Proponen
un protocolo en tres etapas — chequear los archivos que vienen de terreno, chequear la
clasificación, medir el error de lo clasificado — y nueve criterios que todo artículo debería
declarar.

**Dónde queda nuestro trabajo respecto de eso.** Su protocolo cabe **dentro** del nuestro. Las
tres etapas del paper corresponden a fases de nuestra cadena, y nuestra cadena agrega lo que
el paper no cubre: lo que tiene que ser cierto *antes* de que una etiqueta signifique algo
(identidad de la estación, relojes, orden de captura) y lo que tiene que seguir siendo cierto
*después* (el denominador de esfuerzo, las banderas de validez, y el contrato con los
proyectos que usan los datos).

**La diferencia de fondo.** El paper especifica **qué hay que declarar**. Nuestra cadena
**produce cada declaración como un registro**: una columna en la tabla publicada, un test que
falla si la regla se rompe, o un campo del contrato. Lo que el paper pide reportar no se
reconstruye de memoria al escribir un artículo; existe desde el momento en que se tomó la
decisión. Eso toca una parte del problema que el paper documenta: si casi nadie reporta cómo
corrigió sus relojes, una razón probable es que casi nadie lo dejó registrado.

**Dónde estamos por debajo.** En una cosa, y la decimos de entrada: la **tasa de error medida**
sobre una muestra re-revisada a ciegas, que es el centro de la tercera etapa del paper. No la
tenemos porque requiere un segundo revisor y hoy hay uno. Está aplazada, no descartada (§6).

| Etapa del paper | Cubierta hoy | Lo que falta |
|---|---|---|
| (a) Archivos de terreno | 7 de 8 | Estampar el código de sitio sobre la imagen |
| (b) Clasificación | 6 de 9 | Doble revisión (⏸); conteo de individuos, sexo y edad |
| (c) Control de calidad | 3 de 6 | Tasa de error medida, y los dos criterios que dependen de ella (⏸) |

---

## 2. Qué proponen Silva-Rodríguez et al.

### 2.1 El diagnóstico

La revisión de 147 artículos (Web of Science, muestra aleatoria, 2021–2023) mide qué
proporción reporta cada elemento necesario para evaluar la calidad de los datos:

| Criterio de reporte | % que lo cumple | IC 95 % |
|---|---|---|
| Esfuerzo de muestreo | 94,6 % | 89,6–97,2 |
| Criterios de clasificación | 48,3 % | 40,4–56,3 |
| Software de clasificación | 28,6 % | 21,9–36,3 |
| Criterios de exclusión de cámaras | 23,1 % | 17,0–30,6 |
| Número de revisores | 11,6 % | 7,3–17,7 |
| Número de revisiones | 10,2 % | 6,3–16,2 |
| Experiencia de los revisores | 7,5 % | 4,2–12,9 |
| **Métricas de control de calidad** | **4,8 %** | **2,3–9,5** |
| Corrección de errores de programación | 0,7 % | 0,1–3,8 |

> *Nota de lectura:* el texto del artículo dice «10,2 % reportó el número de revisores»
> mientras su Tabla 1 asigna 11,6 % a revisores y 10,2 % a número de revisiones. Se usa la
> tabla. Antes de citar cualquiera de estas cifras fuera de este documento, conviene
> verificarla contra el original.

La conclusión de los autores es que este silencio **induce el supuesto implícito de que las
bases de fototrampeo están libres de error**, supuesto que casi nunca se cumple.

### 2.2 El protocolo

Tres etapas, cada una con su lista de chequeos:

- **(a) Archivos que vienen de terreno.** Que las imágenes de una tarjeta no se atribuyan a la
  cámara equivocada; que la fecha y hora estén bien configuradas; que se registre la hora de
  instalación y retiro de forma independiente; que se detecte el mal funcionamiento de la
  cámara y se declare cómo se excluye.
- **(b) Clasificación.** Definir categorías y criterios antes de clasificar; no forzar la
  identificación a especie; declarar si la clasificación es manual, automática o mixta, con
  modelo, versión y métricas de la herramienta; declarar revisores y revisiones.
- **(c) Control de calidad sobre lo clasificado.** Re-revisar una muestra aleatoria, estimar la
  tasa de error con su intervalo de confianza, compararla contra un umbral de aceptación
  declarado, y revisar el 100 % de los registros de especies raras.

Su pregunta es, en esencia, **si la etiqueta puesta sobre cada imagen es la correcta, y si un
lector puede comprobarlo**.

---

## 3. Cómo encaja su protocolo en el nuestro

Nuestra cadena tiene once fases, que se agrupan en tres bloques. Las etapas del paper caen
dentro del bloque central y de una parte del primero; el resto de la cadena protege cosas que
el paper no aborda.

| Bloque | Fases del manual | Qué protege | Etapa del paper que cae aquí |
|---|---|---|---|
| **Antes de clasificar** | 0 Identidad de estaciones · 1 Visita a terreno · 2 Ingreso del registro de terreno · 3 De la tarjeta al almacenamiento | Que cada imagen se atribuya a la estación, la visita y la tarjeta correctas, y que el orden de captura llegue intacto | (a): almacenamiento, horas de instalación y retiro, mal funcionamiento |
| **Clasificar** | 4 Detección y revisión humana · 5 Revisión de especies y resolución | Que cada imagen con animal llegue a ojos humanos, y que cada veredicto diga de dónde salió | (b) completa; (c) en lo que no depende de un segundo revisor |
| **Después de clasificar** | 6 Diagnóstico del reloj · 7 Anclaje y reparación · 8 Tabla canónica · 9 Contrato · 10 Control de admisión del consumidor | Que la hora sea la verdadera o esté marcada como no confiable, que el denominador de esfuerzo sea el real, y que ningún proyecto aguas abajo lea datos distintos de los publicados | (a): errores de fecha y hora — el paper los previene en terreno; nosotros además los diagnosticamos y reparamos sobre los archivos |

Dos observaciones sobre esta tabla.

**La pregunta del paper es una pregunta de nuestra cadena, no una pregunta paralela.** «¿Es
correcta esta etiqueta?» es exactamente lo que resuelven las Fases 4 y 5. Lo que nuestra cadena
agrega es que esa pregunta sólo tiene sentido si antes se sabe que la imagen es de la estación
que dice y del momento que dice, y si después la etiqueta llega intacta a quien la analiza.
Una base con identificaciones perfectas y relojes corridos ocho años es inutilizable; una
etiqueta correcta en una estación mal atribuida es un dato falso con una especie verdadera.

**El bloque de después no tiene contraparte en el paper.** Los incidentes que lo motivaron
están en §5. Son fallas que ningún control de clasificación puede ver, porque ocurren cuando
la clasificación ya terminó.

---

## 4. Criterio por criterio

Para cada uno de los nueve criterios de reporte: dónde vive en nuestra cadena y **qué registro
lo produce** — la columna, el archivo o la declaración que hace que la respuesta exista sin
tener que reconstruirla.

Estado: **✅ cubierto** · **⏸ aplazado con razón** · **❌ no cubierto**

| # | Criterio del paper | Estado | Dónde vive | Qué registro lo produce | Qué lo protege |
|---|---|---|---|---|---|
| 1 | **Esfuerzo de muestreo** | ✅ | Fase 8 | `valid_effort` por fila; ventana de instalación por estación en `deployments.csv`; estaciones desplegadas sin imágenes publicadas con `has_media = false` | `TestWindowIsTheFieldWindow`, `test_every_station_with_images_has_a_window`, `test_stations_deployed_without_images_are_published` |
| 2 | **Criterios de exclusión de cámaras** | ✅ | Fases 6 y 8 | No excluimos: marcamos. `repair_method` nombra la razón de cada fila; `media_status` da la razón de un vacío y no sólo el hecho | `TestMediaStatusIsAReasonNotAMeasurement` (6 casos), `TestConsumerGuard` |
| 3 | **Criterios de clasificación** | ✅ | Fase 5 | Precedencia explícita R1–R5; `species.yaml` como único vocabulario | `NamedSpeciesWins`, `NegationWins`, `SpeciesFromComment`, `CoarseAndNoteComments`, `FailClosed` |
| 4 | **Software de clasificación** | ✅ | Fases 4 y 5 | Modelo, versión y umbrales declarados por campaña (§4F.2 del manual) | `TestReadTotalExport`, `TestStillsOnly` |
| 5 | **Corrección de errores de programación** | ✅ | Fases 6 y 7 | `repair_method` por fila: reparable, acotable o irreparable, con la razón | 40 tests en 18 clases (`test_clocks.py`), más `test_anchors.py` y `test_timestamps.py` |
| 6 | **Número de revisores** | ✅ | Fase 5 | Declarado por campaña (§5F.3 del manual) | Fila de vigilancia en §5F |
| 7 | **Experiencia del revisor** | ✅ | Fase 5 | §5F.3 declara número y rol; el nombre y la experiencia van en la sección de métodos de cada artículo | — |
| 8 | **Número de revisiones** | ✅ | Fases 4 y 5 | Una pasada de detección y una de especies por campaña. Las pasadas de revisión posteriores **no** son campañas y no entran como tales | `test_retired_campaign_is_not_in_the_published_state` |
| 9 | **Métricas de control de calidad** | ✅ / ⏸ | Fase 5 | `review_outcome` por imagen: `confirmed` o `corrected`. La re-revisión ciega está aplazada | `TestSchemaIsTheContract`; ver §6 |

### 4.1 Lo que dice, y lo que no dice, el criterio 9

`review_outcome` registra **por imagen** si el revisor confirmó o corrigió la propuesta
automática:

| Campaña | `corrected` | `confirmed` | % corregido |
|---|---|---|---|
| otoño 2025 | 540 | 290 | 65 % |
| primavera 2025 | 450 | 294 | 60 % |
| otoño 2026 | 922 | 863 | 52 % |
| **Total** | **1.912** | **1.447** | **57 %** |

Es evidencia fuerte de que la revisión humana es sustantiva y no una confirmación en bloque: si
lo fuera, la proporción sería la inversa. **Pero no es una tasa de error.** Mide el acuerdo
entre la propuesta automática y el revisor, no el error que el revisor deja. §6 explica por qué
la distinción importa.

### 4.2 Las tres etapas, chequeo por chequeo

**Etapa (a) — Archivos de terreno**

| Lo que el paper pide | Estado | Nuestra cobertura | Tests |
|---|---|---|---|
| Que las imágenes de una tarjeta no se atribuyan a la cámara equivocada | ✅ | Fase 3: una carpeta por estación en el nivel superior; una carpeta anidada es precondición fatal, no advertencia | `TestFindNestedStations`, `TestNamesAStation`, `TestShapeOf` |
| ID único por tarjeta ligado a sitio y período | ✅ ▲ | Fase 3: hoja de registro de tarjetas. Además, la **posición** de cada sitio viaja verificada: el contrato publica el hash del registro de estaciones y toda estación observada tiene que estar en él | `test_every_observed_station_is_in_the_registry`, `test_regenerated_registry_without_republish_is_caught` |
| Estampar el código de sitio sobre la propia imagen | ❌ | No lo hacemos. La atribución se protege por estructura, no en la imagen | — |
| Errores de fecha y hora | ✅ ▲ | Fases 6 y 7 completas (§5, punto 1) | 40 tests en 18 clases |
| Fotografiar instalación y retiro; revisar primer y último archivo | ✅ | Fase 1: gatillar la cámara en cada visita (§1F.4) y revisar el primer y último archivo contra la fila de la visita (§1F.7) | `TestTheRule` (12 casos), `TestDeploymentWindow` |
| Registrar independientemente la hora de instalación y retiro | ✅ | Fase 1: dos lecturas crudas de reloj, y **ningún campo de veredicto** por diseño | `TestSchema`, `TestTheFormsObligations` |
| Mal funcionamiento de la cámara | ✅ | §1F.7 y tres columnas del formulario de visita (ver abajo) | `TestTheMalfunctionChecks` (7 casos), `TestTheRecordRefusesAnOldShape` |
| Declarar criterios de exclusión por mal funcionamiento | ✅ | `media_status`, `has_media`, `valid_effort` | `TestMediaStatusIsAReasonNotAMeasurement` |

*Sobre el mal funcionamiento.* Fue el único punto donde el paper cubría algo que no teníamos, y
el único donde la comparación agregó dato nuevo. Una cámara mal apuntada durante toda una
campaña no rompe ningún control: no pierde archivos, no falla relojes, no descuadra conteos;
sólo baja la tasa de detección en silencio. Desde el 2026-09-09 el formulario pregunta
`aim_intact` en cada visita, `stop_reason` (siete razones, «no se sabe» incluida) cuando la
cámara no funcionaba, y `last_known_working` cuando se puede establecer. Son tres columnas y no
los ocho puntos del paper porque los ocho colapsan a tres consecuencias analíticas, y ocho
casillas se contestan mecánicamente.

**Etapa (b) — Clasificación**

| Lo que el paper pide | Estado | Nuestra cobertura | Tests |
|---|---|---|---|
| Definir categorías y criterios antes de clasificar | ✅ | `species.yaml` es el vocabulario único; la vía «Otro (especificar)» sólo resuelve a través de él | `SpeciesFromComment` |
| No forzar identificación a especie en registros dudosos | ✅ | R4 deja `unknown` etiquetado por tipo en vez de promover a especie. Falta la instrucción explícita al revisor | `CoarseAndNoteComments` |
| Declarar el enfoque (manual / automático / mixto) | ✅ | Mixto: MegaDetector detecta, CLIP propone, una persona decide por imagen | `TestSchemaIsTheContract` |
| Declarar modelo, **versión** y métricas de la IA | ✅ | MegaDetector v5. Se evaluaron las versiones v5a y v5b comparando sus curvas de precisión-recall (virtualmente idénticas) y se fijó el umbral en 0,38 → recall 0,97 / precisión 0,80. Por campaña: `md_v5b.0.0` en otoño 2025 y otoño 2026, `MD5A-0-0` en primavera 2025 | §4F.2 del manual |
| Criterio para contar individuos (por imagen o por secuencia) | ❌ | No contamos individuos. Sí agrupamos eventos: `episode_30min` | `TestTheGapIsMeasuredFromTheLastRetainedDetection`, `TestAnEpisodeCannotCrossAClockSegment` |
| Sexo, clase de edad, asociación con personas | ❌ | No capturado. La asociación perro-persona sería útil para Bosque Pehuén | — |
| Doble revisión y resolución de discrepancias | ⏸ | Un solo revisor (§6) | — |
| Que exactamente un archivo sea el registro revisado | ✅ ▲ | Fase 5. Regla nuestra que el paper no tiene | `RowSetIsTheExport` |
| Registrar de dónde salió cada veredicto | ✅ ▲ | `review_resolution` es columna publicada | `WhereTheVerdictComesFrom` (6 casos) |

**Etapa (c) — Control de calidad sobre lo clasificado**

| Lo que el paper pide | Estado | Nuestra cobertura |
|---|---|---|
| Muestra aleatoria re-revisada para estimar la tasa de error, con IC de Wilson | ⏸ | Aplazado: no hay segundo revisor (§6) |
| Umbral de aceptación declarado (su ejemplo: 0,5 %) | ⏸ | Sin tasa medida no hay umbral que declarar |
| Precisión y recall por clase | ⏸ | Aplicarían al paso humano, no al detector (§6) |
| Revisar el 100 % de los registros de especies raras | ✅ | El revisor abre cada imagen con detección de animal, lo que incluye por construcción todas las especies raras |
| Que el control lo haga quien es responsable de la base | ✅ | Un solo responsable de la base |
| Reconocer quién hizo cada etapa | ✅ | Quién clasificó cada campaña queda registrado |

---

## 5. Lo que nuestra cadena agrega

Una forma útil de leer esta sección: supongamos un artículo que cumple al pie de la letra el
protocolo de Silva-Rodríguez et al. **¿Qué seguiría sin poder comprobar su lector?** Cada punto
de abajo es una respuesta, y cada uno nació de una falla que nos costó datos. No es una crítica
al paper — es un protocolo general, y éstos son problemas específicos que encontramos al
operarlo.

**1. Si la hora de cada registro es la verdadera.** El paper trata los errores de fecha y hora
en un párrafo. Nosotros tenemos nueve clases de error de reloj, diagnóstico **por segmento**,
dos precondiciones que rechazan en vez de adivinar, y una regla de reparación de una línea:
*un segmento es reparable si y sólo si es coherente y contiene al menos un ancla*. Cada fila
dice si su hora es **reparada, acotada o irreparable**, y por qué, en `repair_method`.
→ 40 tests en 18 clases.

**2. Si se perdió algo entre la tarjeta y el disco.** Prohibimos aplanar, renombrar o
deduplicar por nombre de archivo. Dos archivos con el mismo nombre son exactamente lo que
produce una cámara reiniciada; borrar uno como «duplicado» destruye la evidencia de la falla y
baja el conteo, de modo que la pérdida es invisible. El orden de captura se trata como
evidencia.
→ `TestPrefixCandidates`, `TestResolveDest`, `TestTheManifestDescribesEveryFile`,
`TestTheManifestIsADeletionLedger`.

**3. Qué se descartó, y si había que descartarlo.** El paper habla de excluir cámaras y
declarar el criterio. Nosotros no excluimos: publicamos con banderas. Ninguna de las 4.094
filas con reloj no limpio se descartó; están en la tabla con `valid_time_of_day = false`,
disponibles para preguntas de presencia y excluidas de preguntas de hora. Excluirlas habría
botado presencia perfectamente válida.

**4. Para qué pregunta sirve cada registro.** Tres ejes de validez independientes —
`valid_date`, `valid_time_of_day`, `valid_effort` — en vez de una sola bandera. Un error puro
de año preserva la hora del día exactamente; una bandera única botaría el registro de
actividad de una cámara cuyo único defecto es el calendario.
→ `TestConsumerGuard`.

**5. Si el denominador es el real.** El paper pide reportar el esfuerzo; no dice cómo el
esfuerzo se corrompe. A nosotros se nos corrompió: video guardado fuera del árbol de la
campaña hizo que cuatro estaciones que grabaron todo el tiempo se leyeran como cámaras que no
vieron nada, unos 500 días-cámara mal clasificados. Desde entonces `media_status` da la
**razón** de un vacío, y las estaciones desplegadas sin imágenes se publican con
`has_media = false`.
→ `TestMediaStatusIsAReasonNotAMeasurement` (6 casos).

**6. Si algo se ignoró en silencio.** Un comentario de revisión que el sistema no reconoce
**detiene el ingreso**. No se ignora ni se adivina. Tratar lo desconocido como vacío produce un
conjunto más chico sin mensaje de error, que es el peor resultado posible porque se parece al
éxito.
→ `FailClosed` (5 casos).

**7. Si el análisis usó los mismos datos que se publicaron.** La tabla canónica se publica con
un contrato (`CANONICAL_STATE.json`) y cada proyecto que la usa lo verifica antes de leer. El
contrato cubre filas, columnas, días-cámara y, desde el esquema 5 (2026-10-05), el **registro
de estaciones**: hasta entonces cada coordenada y cada altitud cruzaba sin verificar. Nada de
esto está en el paper, y nos costó tres incidentes: un lector aguas abajo que rederivó cinco
decisiones del productor y discrepaba en 515 filas; un proyecto que cargó una pasada de
revisión en vez de la campaña y movió las detecciones de liebre de 230 a 161; un tercero que
reinterpretó etiquetas de estación en tres gramáticas distintas.
→ `TestSchemaIsTheContract`, `TestConsumerGuard`, `TestPublishedFileIsCurrent`,
`TestDiffDetectsRealChanges`, `test_every_observed_station_is_in_the_registry`.

**8. Si dos análisis cuentan los eventos igual.** `episode_30min` se calcula una vez, en el
origen, y no puede cruzar un segmento de reloj. Antes de eso, dos implementaciones aguas abajo
discrepaban en un tercio de los eventos. El paper discute el conteo por secuencia sin decir
dónde debe vivir esa regla.
→ `TestAnEpisodeCannotCrossAClockSegment`, `TestOrderIndependence`, `TestWhatGetsNoEpisode`.

**9. Si las reglas se siguen cumpliendo.** El paper propone un protocolo en prosa. Las nuestras
son ejecutables: cada regla del manual tiene una clase de tests que falla visiblemente el día
que la regla se rompe (el conteo vigente está en §0.6 del manual). Una convención escrita se
degrada; una ejecutable no.

---

## 6. Dónde estamos por debajo del estándar, y por qué

**La tasa de error medida — aplazada.** Es el elemento central de la etapa (c): re-revisar a
ciegas una muestra aleatoria y estimar el error con su intervalo. El paper sugiere ~5.000
archivos; para 35.807 filas y tres campañas, una versión reducida (300–400 imágenes
estratificadas) bastaría. Está aplazada por una razón operativa, no por desacuerdo: **hoy
nadie más del equipo revisa imágenes**, y una muestra re-revisada por la misma persona que
clasificó no mide error, mide consistencia consigo misma. Es la primera mejora a activar
cuando haya un segundo revisor, y con ella la doble revisión.

**Por qué el 57 % no la reemplaza.** Mide acuerdo humano-máquina, no error residual. Cualquier
frase que lo convierta en «nuestra base tiene un X % de error» es falsa, y un revisor
competente la va a encontrar.

**Precisión y recall por especie — no aplican donde el paper las pone.** MegaDetector no
clasifica especies: sólo detecta animales. Las métricas por clase aplicarían al paso humano, y
dependen de la re-revisión ciega.

**El umbral del detector, con sus dos límites.** Se eligió 0,38 (recall 0,97, precisión 0,80)
privilegiando la exhaustividad: una imagen marcada de más sólo cuesta tiempo de revisor,
mientras que una imagen no marcada **no la mira nadie nunca** y desaparece sin dejar señal.
Dos límites que hay que enunciar junto al número: el 0,97 es una propiedad del detector medida
sobre un conjunto de evaluación, no sobre todas nuestras imágenes; y el 3 % que se pierde no es
parejo — los fallos de un detector se concentran en animales chicos, lejanos, nocturnos en
infrarrojo y parcialmente ocluidos, lo que introduce detectabilidad diferencial entre taxones y
entre estaciones. No invalida nada; se declara como limitación conocida.

**Lo que no capturamos.** Conteo de individuos, sexo, clase de edad y asociación con personas.
Tampoco estampamos el código de sitio sobre la imagen; lo reemplazamos por protección
estructural.

**Un pendiente menor, anotado.** Las imágenes sin detección de animal llevan `review_outcome`
vacío, que hoy significa siempre «no aplica» pero podría leerse como «no se revisó». El cambio a
`not_applicable` es una modificación de la tabla publicada y se hará como operación aparte.

---

## 7. Estado

| Etapa del paper | Cubierta | Más allá del paper (▲) | Queda fuera |
|---|---|---|---|
| (a) Archivos de terreno | 7 de 8 | Reloj diagnosticado y reparado por segmento; posición de la estación verificada en el contrato | Estampado del código sobre la imagen |
| (b) Clasificación | 6 de 9 | Un solo archivo revisado; procedencia de cada veredicto | Doble revisión (⏸); individuos, sexo y edad (❌) |
| (c) Control de calidad | 3 de 6 | — | Tasa de error medida y los dos criterios que dependen de ella (⏸) |
| *Sin contraparte en el paper* | — | Denominador, banderas de validez, eventos, contrato con los consumidores (§5) | — |

Las filas se leen distinto y conviene no promediarlas. La etapa (a) está prácticamente
completa. La etapa (b) está cubierta en todo lo que depende del proceso; lo que falta son
datos que no capturamos. La etapa (c) está corta por una restricción de personal, no de método.

---

## 8. Potencial

Lo que este documento muestra es que un estándar publicado para la calidad de datos de
fototrampeo cabe dentro de nuestra cadena, y que la cadena resuelve además una parte del
problema que ese estándar documenta pero no aborda: que la información necesaria para confiar
en los datos **exista como registro**, y no dependa de que alguien la reconstruya al escribir.

Eso tiene potencial como contribución metodológica, y vale la pena tenerlo presente. Antes de
afirmarlo fuera del equipo hacen falta dos cosas: una revisión de literatura que este
documento no hizo — como mínimo frente a estándares de intercambio de datos como Camtrap DP
(Bubnicki et al., 2023) y a las plataformas de gestión de fototrampeo —, y la tasa de error
medida (§6), sin la cual un revisor externo encontraría el vacío antes que la contribución.
