# Nuestro protocolo frente a Silva-Rodríguez et al. (2025)

**Fundación Mar Adentro · Bosque Pehuén**
Documento de trabajo. Edición 2026-09-09.
Las cuatro correcciones que este documento proponía **ya están aplicadas** en
`MANUAL-SALUD-DATOS.md` (edición 2026-09-09) y en el código; los estados de abajo
reflejan la cadena tal como está hoy.

> Silva-Rodríguez, E. A., Cortés, E. I., Vasquez-Ibarra, V., Gálvez, N., Cusack, J.,
> Ohrens, O., Moreira-Arce, D., Farías, A. A. & Infante-Varela, J. (2025).
> *A protocol for error prevention and quality control in camera trap datasets.*
> Journal of Applied Ecology. DOI: 10.1111/1365-2664.70010

---

## 1. Qué es este documento, y qué no es

Es un **mapa de correspondencia** entre el protocolo publicado por Silva-Rodríguez et al. y
la cadena de datos de Bosque Pehuén, con los tests que sostienen cada punto. Existe para
responder dos preguntas concretas:

1. De todo lo que ese protocolo pide **explicitar y fundamentar**, ¿qué tenemos, dónde vive,
   y qué lo protege de perderse?
2. ¿Dónde fuimos más allá de lo que el protocolo exige, y por qué valió la pena?

**Qué no es, y conviene fijarlo antes de seguir porque son tres registros distintos:**

| Documento | Para quién | Qué pregunta responde |
|---|---|---|
| `MANUAL-SALUD-DATOS.md` | El equipo, en terreno y en gabinete | *¿Cómo se hace, y qué se rompe si no se hace así?* |
| Declaración de métodos (futura) | Revisores y lectores de un artículo | *¿Qué hicieron ustedes, y por qué debería creerles?* |
| **Este documento** | Nosotros, al planificar | *¿Cómo se compara lo nuestro con el estándar publicado?* |

No es el manual: no enseña a nadie a hacer nada. No es la sección de métodos: no está escrito
para un lector externo ni tiene el tono de una declaración. Es el insumo del que después salen
las dos cosas.

---

## 2. Las dos preguntas distintas

El hallazgo central de la comparación es que **los dos protocolos vigilan cosas
complementarias y casi no se superponen**:

- **Silva-Rodríguez et al. protegen la clasificación.** Su pregunta es *"¿la etiqueta puesta
  sobre esta imagen es la correcta?"*. Su respuesta es medir: muestrear, re-revisar, calcular
  tasa de error, precisión y recall, y comparar contra un umbral.

- **Nuestro manual protege la procedencia.** Su pregunta es *"¿este dato viene de donde dice
  que viene, en el momento que dice, y con qué denominador?"*. Su respuesta es rechazar
  temprano y de forma visible, con precondiciones deterministas y tests ejecutables.

Ninguna de las dos sustituye a la otra. Una base con identificaciones perfectas y relojes
corridos ocho años es inutilizable; una base con relojes impecables y especies mal
identificadas también. Lo que sigue es el cruce completo.

### El contexto que da peso al ejercicio

Los autores revisaron **147 artículos** de fototrampeo publicados entre 2021 y 2023 (Web of
Science, muestra aleatoria) y midieron qué proporción reportaba cada criterio:

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
> tabla. Si alguna de estas cifras se cita en un artículo nuestro, conviene verificarla contra
> el original.

La conclusión de los autores es que el silencio sobre control de calidad **induce el supuesto
implícito de que las bases de fototrampeo están libres de error**, supuesto que casi nunca se
cumple. Ése es el estándar contra el que nos estamos midiendo.

---

## 3. Los nueve criterios de reporte, uno por uno

Estado: **✅ cubierto** · **⏸ aplazado con razón** · **❌ no cubierto**

| # | Criterio del paper | Estado | Dónde vive en nuestra cadena | Qué lo protege |
|---|---|---|---|---|
| 1 | **Esfuerzo de muestreo** | ✅ | Fase 8: `valid_effort`, ventana de instalación por estación, estaciones publicadas con `has_media = false` | `TestWindowIsTheFieldWindow`, `test_every_station_with_images_has_a_window`, `test_stations_deployed_without_images_are_published` |
| 2 | **Criterios de exclusión de cámaras** | ✅ | Fase 6 y 8: no excluimos, marcamos. `repair_method` nombra la razón de cada fila; `media_status` da la razón y no sólo el hecho | `TestMediaStatusIsAReasonNotAMeasurement` (6 casos), `TestConsumerGuard` |
| 3 | **Criterios de clasificación** | ✅ | Fase 5: precedencia explícita R1–R5, `species.yaml` como único vocabulario | `NamedSpeciesWins`, `NegationWins`, `SpeciesFromComment`, `CoarseAndNoteComments`, `FailClosed` |
| 4 | **Software de clasificación** | ✅ | Fase 4 y 5: MegaDetector v5 + CLIP zero-shot + revisión humana en `phase1_labeling` | `TestReadTotalExport`, `TestStillsOnly` |
| 5 | **Corrección de errores de programación** | ✅ | Fases 6 y 7 completas: nueve clases de error de reloj, precondiciones, anclaje | 40 tests en 18 clases (`test_clocks.py`) + `test_anchors.py`, `test_timestamps.py` |
| 6 | **Número de revisores** | ✅ | Un revisor por campaña. Queda registrado en el manual y en el registro de campaña | §5F.3, más su fila de vigilancia |
| 7 | **Experiencia del revisor** | ✅ | §5F.3 declara número y rol del revisor. El descriptor de experiencia va en la declaración de métodos, no en el manual | — |
| 8 | **Número de revisiones** | ✅ | Una pasada de detección (Fase 4) y una de especies (Fase 5), más las pasadas de revisión posteriores, que **no** son campañas | `test_retired_campaign_is_not_in_the_published_state` |
| 9 | **Métricas de control de calidad** | ✅ / ⏸ | Tasa de corrección humana sobre la propuesta automática: **1.912 `corrected` contra 1.447 `confirmed`** sobre las tres campañas (57 % corregido). Publicada en la columna canónica `review_outcome`. La re-revisión ciega queda aplazada | `TestSchemaIsTheContract` (declara la columna); ver §6 |

**Sobre el criterio 9, y hay que decirlo con precisión porque es fácil sobrevenderlo.** El
57 % mide el **acuerdo entre la propuesta automática y el revisor**, no el error residual del
revisor. Es evidencia fuerte de que la revisión humana es sustantiva y no una confirmación en
bloque —si hubiera sellos de goma la proporción sería la inversa— y es una métrica de
transparencia legítima para el paso semi-automatizado. **No es una tasa de error del dataset y
no debe presentarse como tal.**

Desglose por campaña:

| Campaña | `corrected` | `confirmed` | % corregido |
|---|---|---|---|
| otoño 2025 | 540 | 290 | 65 % |
| primavera 2025 | 450 | 294 | 60 % |
| otoño 2026 | 922 | 863 | 52 % |
| **Total** | **1.912** | **1.447** | **57 %** |

Las 32.448 filas restantes de las 35.807 publicadas tienen `review_outcome` vacío: son las
imágenes sin detección de animal sobre el umbral, que nunca entraron a la interfaz de
especies. **Pendiente (no aplicado):** ese vacío debería decir `not_applicable`, porque hoy un valor
vacío en una columna publicada significa dos cosas distintas — «no aplica» y «no se revisó» —
que es exactamente el modo de falla que §4F.3 del manual identifica como el más difícil de ver.

---

## 4. El marco de tres etapas del paper, contra nuestras once fases

### Etapa (a) — Chequeos de los archivos que vienen de terreno

| Lo que el paper pide | Estado | Nuestra cobertura | Tests |
|---|---|---|---|
| **Errores de almacenamiento**: que las imágenes de una tarjeta no se atribuyan a la cámara equivocada | ✅ | Fase 3: una carpeta por estación en el nivel superior; carpeta anidada es precondición fatal, no advertencia | `TestFindNestedStations`, `TestNamesAStation`, `TestShapeOf` |
| ID único por tarjeta ligado a sitio y período | ✅ | Fase 3: hoja de registro de tarjetas | — |
| Estampar el código de sitio sobre la propia imagen | ❌ | No lo hacemos. La atribución se protege estructuralmente, no en la imagen | — |
| **Errores de programación** (fecha y hora mal configuradas) | ✅ ▲ | Fases 6 y 7 completas. Muy por encima de lo que el paper pide | 40 tests en 18 clases |
| Fotografiar instalación y retiro; revisar primer y último archivo | ✅ | Fase 1 exige gatillar la cámara en cada visita (§1F.4); §1F.7 agrega la revisión del primer y último archivo contra la fila de la visita | `TestTheRule` (12 casos), `TestDeploymentWindow` |
| Registrar independientemente la hora de instalación y retiro | ✅ | Fase 1: dos lecturas crudas de reloj, y **ningún campo de veredicto** por diseño | `TestSchema`, `TestTheFormsObligations` |
| **Mal funcionamiento de cámara** | ✅ | §1F.7, más tres columnas del formulario: `aim_intact`, `stop_reason`, `last_known_working` | `TestTheMalfunctionChecks` (7 casos), `TestTheRecordRefusesAnOldShape` |
| Declarar criterios de exclusión por mal funcionamiento | ✅ | `media_status`, `has_media`, `valid_effort` | `TestMediaStatusIsAReasonNotAMeasurement` |

### Etapa (b) — Chequeos de la clasificación de la base

| Lo que el paper pide | Estado | Nuestra cobertura | Tests |
|---|---|---|---|
| Definir categorías y criterios de clasificación primero | ✅ | `species.yaml` es el vocabulario único; la vía «Otro (especificar)» sólo resuelve a través de él | `SpeciesFromComment` |
| No forzar identificación a especie en registros dudosos | ✅ | R4 deja `unknown` etiquetado por tipo en vez de promover a especie. Falta la instrucción explícita al revisor | `CoarseAndNoteComments` |
| Declarar el enfoque (manual / automático / semi-automático) | ✅ | §5F.3: MegaDetector detecta, CLIP propone, un humano decide por imagen | `TestSchemaIsTheContract` |
| Declarar modelo, **versión** y métricas de la IA | ✅ | MegaDetector v5, evaluado sobre v5a y v5b (curvas virtualmente idénticas), umbral 0,38 → recall 0,97 / precisión 0,80. Campañas: `md_v5b.0.0` en otoño 2025 y otoño 2026, `MD5A-0-0` en primavera 2025 | §4F.2 |
| Criterio para el conteo de individuos (por imagen o por secuencia) | ❌ | No contamos individuos. Sí agrupamos eventos: `episode_30min` | `TestTheGapIsMeasuredFromTheLastRetainedDetection`, `TestAnEpisodeCannotCrossAClockSegment` |
| Sexo, clase de edad, asociación con personas | ❌ | No capturado. La asociación perro-persona sería útil para Bosque Pehuén | — |
| Doble revisión y resolución de discrepancias entre revisores | ⏸ | Un solo revisor. Ver §6 | — |
| Que exactamente un archivo sea el registro revisado | ✅ ▲ | Fase 5, y es una regla nuestra que el paper no tiene | `RowSetIsTheExport` |
| Registrar de dónde salió cada veredicto | ✅ ▲ | `review_resolution` es columna publicada | `WhereTheVerdictComesFrom` (6 casos) |

### Etapa (c) — Control de calidad sobre los archivos clasificados

| Lo que el paper pide | Estado | Nuestra cobertura |
|---|---|---|
| Muestra aleatoria re-revisada para estimar tasa de error, con IC de Wilson | ⏸ | Aplazado: no hay segundo revisor. Ver §6 |
| Umbral de aceptación declarado (su ejemplo: 0,5 %) | ⏸ | Sin tasa medida no hay umbral que declarar |
| Precisión y recall por clase, o macro-promediadas | ⏸ | Aplicaría al paso humano, no al detector. Ver §6 |
| Revisar el 100 % de las clasificaciones positivas de especies raras | ✅ | Se hace: el revisor abre cada imagen con detección de animal, lo que incluye por construcción el total de las especies raras |
| Que el control de calidad lo haga quien es responsable de la base | ✅ | Un solo responsable de la base |
| Reconocer quién hizo cada etapa | ✅ | Se agrega el registro de quién clasificó cada campaña |

---

## 5. El chequeo de mal funcionamiento — cerrado el 2026-09-09

Era el único punto donde el paper cubría algo para lo que no teníamos ni regla, ni código, ni
evidencia de que el problema no estuviera ocurriendo. Que era real ya estaba documentado por
accidente, dos veces: una tarjeta SD defectuosa anotada **en el nombre de la carpeta de la
cámara**, y una cámara encontrada **apuntando hacia arriba**, en comentarios. Los dos datos
sobrevivieron por casualidad — nadie los pidió y nadie los buscaría.

Es la clase de falla que **no rompe ningún control existente**: no pierde archivos, no falla
precondiciones de reloj, no descuadra conteos. Sólo baja la tasa de detección de una estación
en silencio.

**Qué se hizo.** Tres columnas nuevas en el formulario de visita, agrupadas por la medición que
protegen, más la lista de ocho puntos en §1F.7 del manual para leer en el sitio:

| Columna | Qué captura | Obligatoria |
|---|---|---|
| `aim_intact` | ¿Apuntaba donde corresponde? Cubre ángulo, altura, vegetación sobre el lente, cámara movida | Siempre |
| `stop_reason` | Vocabulario cerrado de siete razones, `no se sabe` incluida | Sólo si `camera_working = no` |
| `last_known_working` | La fecha de muerte, cuando se puede establecer | Opcional |

Tres decisiones de diseño que vale registrar:

- **Columnas y no comentarios**, porque el propio esquema ya lo tenía escrito: el campo `notes`
  declara que *«si un dato importa para el análisis, pedir una columna en vez de escribirlo
  aquí»*. Los dos casos históricos son exactamente datos que importaban y quedaron en prosa.
- **Tres columnas y no ocho.** Los ocho puntos del paper colapsan a tres consecuencias
  analíticas; ocho casillas se contestan mecánicamente, y un formulario contestado
  mecánicamente deja de ser un dato.
- **`stop_reason` es el testigo de terreno detrás de `media_absence.csv`.** Esa declaración
  decide si los días-cámara de una estación entran a un denominador, y hasta ahora se escribía
  meses después, desde la memoria de alguien, sin ninguna observación que la sostuviera.

**Lo que no se hizo, a propósito.** Los dos casos históricos **no** se migraron a las columnas
nuevas. Extraer un valor estructurado de un comentario escrito para una persona es la misma
reinterpretación silenciosa que el cargador rechaza en el ingreso; si vale la pena promoverlos,
es una edición curada, hecha por una persona, con su razón en `data_flags`.

## 6. Lo que adoptamos con reservas, y por qué

**Tasa de error medida sobre muestra ciega — aplazada.** El paper sugiere ~5.000 archivos;
para 35.807 filas y tres campañas una versión reducida (300–400 imágenes estratificadas)
bastaría. Está aplazada por una razón operativa y no por desacuerdo: **hoy nadie más del equipo
revisa imágenes**, y una muestra re-revisada por la misma persona que clasificó no mide error,
mide consistencia consigo misma. Se retoma cuando haya un segundo revisor.

**Precisión y recall por especie — no aplican donde el paper las pone.** MegaDetector no
clasifica especies: sólo detecta animales. Pedirle métricas por especie es un error de
categoría. Las métricas por clase aplicarían al paso humano, y ésas dependen de la re-revisión
ciega aplazada.

**Doble revisión — no aplicable hoy.** Requiere dos personas. Queda anotado como la primera
mejora a activar si el equipo crece.

**Sobre el umbral de detección, la decisión y su fundamento.** Se adoptó 0,38 (recall 0,97,
precisión 0,80) privilegiando **exhaustividad sobre precisión**: una imagen de baja precisión
se revisa igual en el paso de especies y sólo cuesta tiempo de revisor, mientras que un falso
negativo del detector **elimina la observación de forma permanente y sin dejar señal**. Es la
misma asimetría del §0.4 del manual — un error recuperable elegido por sobre uno irrecuperable.

Dos límites que conviene enunciar junto al número, para que no se lea como más limpio de lo que
es: el 0,97 es una propiedad del detector medida sobre un conjunto de evaluación, no sobre
nuestras imágenes; y el 3 % que se pierde **no es un error parejo** — los fallos de MegaDetector
se concentran en animales chicos, lejanos, nocturnos en infrarrojo y parcialmente ocluidos, lo
que introduce detectabilidad diferencial entre taxones y entre estaciones con distinta densidad
de vegetación. No invalida nada; hay que declararlo como limitación conocida.

---

## 7. Dónde fuimos más allá del protocolo

Nueve puntos donde nuestra cadena garantiza algo que Silva-Rodríguez et al. no piden. No es una
crítica al paper: es un protocolo general y éstos son problemas que nos costaron datos.

**1. El reloj como disciplina completa.** El paper despacha los errores de fecha y hora en un
párrafo («errores de programación», reportados por el 0,7 % de la literatura). Nosotros tenemos
nueve clases de error, diagnóstico **por segmento**, dos precondiciones que rechazan en vez de
adivinar, y una regla de reparación de una línea: *un segmento es reparable si y sólo si es
coherente y contiene al menos un ancla*. Y la distinción que ningún protocolo publicado hace:
**reparable / acotable / irreparable**, con la razón nombrada en `repair_method` de cada fila.
→ 40 tests en 18 clases.

**2. El orden de captura tratado como evidencia forense.** La prohibición de aplanar, renombrar
o deduplicar por nombre de archivo no aparece en el paper. Dos archivos con el mismo nombre son
exactamente lo que produce una cámara reiniciada; borrar uno como «duplicado» destruye la
evidencia de la falla a la que se parece, y baja el conteo, de modo que la pérdida es invisible.
→ `TestPrefixCandidates`, `TestResolveDest`, `TestTheManifestDescribesEveryFile`,
`TestTheManifestIsADeletionLedger`.

**3. Marcar en vez de excluir.** El paper habla de *excluir* cámaras y de declarar el criterio.
Nosotros no excluimos: publicamos con banderas. Ninguna de las 4.094 filas con reloj no limpio
se descartó — están todas en la tabla, con `valid_time_of_day = false`, disponibles para
preguntas de presencia y excluidas de preguntas de hora. Excluir habría botado presencia
perfectamente válida.

**4. Tres ejes de validez independientes, no uno.** `valid_date`, `valid_time_of_day`,
`valid_effort`. Un error puro de año preserva la hora del día exactamente; una sola bandera
botaría un registro completo de actividad de una cámara cuyo único defecto es el calendario.
→ `TestConsumerGuard`.

**5. El denominador como objeto de primera clase.** `media_status` da la **razón** y no sólo el
hecho de que no haya imágenes; estaciones desplegadas sin imágenes se publican con
`has_media = false`. Esto nació de un caso real: video guardado fuera del árbol de la campaña
hizo que cuatro estaciones que grabaron todo el tiempo se leyeran como cámaras que no vieron
nada — unos 500 días-cámara mal clasificados. El paper pide reportar esfuerzo; no tiene nada
sobre cómo el esfuerzo se corrompe.
→ `TestMediaStatusIsAReasonNotAMeasurement` (6 casos).

**6. Fallar cerrado ante vocabulario desconocido.** R5: un comentario que el sistema no conoce
**aborta el ingreso**. No se ignora y no se adivina. La alternativa —tratar lo desconocido como
vacío— produce un conjunto más chico sin mensaje de error, que es el peor resultado posible
porque se parece al éxito.
→ `FailClosed` (5 casos).

**7. Frontera productor/consumidor con contrato verificable.** Fases 9 y 10: la tabla canónica
se publica con un contrato (`CANONICAL_STATE.json`) y el consumidor verifica antes de leer.
Nada de esto está en el paper, y nos costó tres incidentes documentados: un lector aguas abajo
que rederivó cinco decisiones del productor y discrepaba en 515 filas vivas; un proyecto que
cargó una pasada de revisión en vez de la campaña y movió las detecciones de liebre de 230 a
161; un tercero que reinterpretó etiquetas de estación en tres gramáticas distintas.
→ `TestSchemaIsTheContract`, `TestConsumerGuard`, `TestPublishedFileIsCurrent`,
`TestDiffDetectsRealChanges`.

**8. La regla de eventos viaja en la tabla.** `episode_30min` se calcula una vez, aguas arriba,
y no puede cruzar un segmento de reloj. Dos implementaciones aguas abajo ya habían discrepado
un 33 %. El paper discute el conteo por secuencia sin decir dónde debe vivir esa regla.
→ `TestAnEpisodeCannotCrossAClockSegment`, `TestOrderIndependence`, `TestWhatGetsNoEpisode`.

**9. Reglas ejecutables en vez de reglas escritas.** **311 tests en 76 clases, 14 archivos.** La
unidad conceptual es la clase, no el test. El paper propone un protocolo en prosa; una
convención documentada se degrada, una ejecutable falla visiblemente el día que se rompe.

---

## 8. Estado consolidado

| Etapa del paper | Antes del 2026-09-09 | Hoy | Queda fuera |
|---|---|---|---|
| (a) Archivos de terreno | 5 de 8, más 1 parcial | **7 de 8** | Estampado del código sobre la imagen |
| (b) Clasificación | 2 de 9 | **6 de 9** | Doble revisión (⏸); conteo de individuos, sexo y edad (❌) |
| (c) Control de calidad | 2 de 6 | **3 de 6** | Tasa de error medida y los tres criterios que dependen de ella (⏸) |

Las tres filas se leen distinto y conviene no promediarlas. La etapa (a) queda prácticamente
completa, y lo que ganó es trabajo real: tres columnas que antes no se preguntaban. La etapa
(b) mejoró casi enteramente **documentando garantías que ya existían en el código** — el
proceso no cambió, cambió que esté escrito y que tenga fila de vigilancia. La etapa (c) sigue
corta, y sigue corta por una restricción de personal, no de método.

**Lo aplicado el 2026-09-09**, en orden de valor:

1. **Chequeo de mal funcionamiento** — §1F.7 del manual, tres columnas del formulario, cuatro
   filas de vigilancia nuevas, migración del registro vivo (107 filas, 22 → 25 columnas, cero
   valores heredados alterados). Es la única de las cuatro que agrega dato nuevo.
2. **Declaración del detector** — §4F.2: modelo, versión por campaña, los dos umbrales, la
   evaluación y el razonamiento recall-sobre-precisión, con sus dos límites enunciados.
3. **Proceso de revisión de especies** — §5F.3 y §5F.4: qué hace cada pieza, `review_outcome`
   como registro por imagen, la cifra de acuerdo con su advertencia, y la semántica del vacío.
4. **Un revisor declarado** — §5F.3, con el nombre fuera del manual y en la declaración de
   métodos, donde corresponde reconocerlo.

**Dos pendientes anotados, ninguno silencioso:**

- **`review_outcome` vacío → `not_applicable`.** Es una modificación de la tabla canónica y
  arrastra la republicación del contrato y a los consumidores detrás. Se documentó la semántica
  en §5F.4; el cambio de valor es una operación aparte.
- **Re-revisión ciega.** Aplazada mientras no haya un segundo revisor.

De paso, dos defectos que la revisión de diseño encontró y que no estaban en el plan:

- **`visit_form.py` nombraba columnas literalmente** (`elif column == 'visit_date'`) pese a que
  su propio docstring promete que no lo hace. Ahora despacha por el formato que declara el
  esquema, así que la próxima columna de fecha no lo toca.
- **El cargador no verificaba la forma del registro.** Escribe con las columnas del formulario y
  sólo emite encabezado si el archivo es nuevo, de modo que cargar sobre un registro de otra
  forma habría archivado cada valor bajo el nombre equivocado, dejando un CSV que aún parecía
  válido. Ahora rechaza. → `TestTheRecordRefusesAnOldShape`.

## 9. Qué se cita de este documento, y qué no

Para una sección de métodos sirven: §3 (los nueve criterios, como checklist de lo que hay que
declarar), §4 (el mapa de etapas), §6 (la fundamentación del umbral, con sus dos límites
declarados) y §7 (los resguardos que exceden el estándar).

**No sirve el 57 % de corrección presentado como tasa de error.** Es acuerdo humano-máquina.
Cualquier frase que lo convierta en «nuestra base tiene un error de X %» es falsa y un revisor
competente la va a encontrar.
