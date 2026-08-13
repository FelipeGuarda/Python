# Formulario de contactos — plantilla FMA

Plantilla única para registrar asistentes a Encuentros y sumarlos a
`LISTADO_CONTACTOS_MAESTRO`. **Duplica este formulario para cada evento; no
crees uno nuevo desde cero.** Cada formulario nuevo reinventa las columnas, y
reinventar las columnas es exactamente lo que obliga a revisar a mano.

Sus columnas alimentan directamente `scripts/merge_contacts.py`. Si se
respetan, el paso de revisión deja de requerir criterio humano y pasa a ser
un trámite.

---

## Configuración del formulario

| Ajuste | Valor | Por qué |
|---|---|---|
| Recopilar direcciones de correo | **Verificada** | Toma la dirección de la cuenta Google, sin errores de tipeo. Es identidad, no destino — por eso igual se pregunta el correo de contacto. |
| Limitar a 1 respuesta | **No** | Exigir cuenta Google para responder excluye gente. |
| Editar después de enviar | **Sí** | Permite corregir un correo mal escrito sin duplicar la fila. |
| Barra de progreso | Sí | Son 6 preguntas; verlo reduce abandono. |
| Mezclar el orden | **No** | El orden de columnas debe coincidir con la maestra. |

**Título:** `Encuentros de Conservación — registro de contacto`

**Descripción:**

> Completa estos datos para recibir invitaciones a nuestros próximos
> Encuentros. Toma menos de un minuto.
>
> Fundación Mar Adentro trata tus datos sólo para invitarte a estas
> instancias. Puedes pedirnos que te demos de baja cuando quieras
> escribiendo a [correo de contacto].

---

## Preguntas

### 1. Nombre — *Respuesta corta, obligatoria*

> **Nombre**

*Sin texto de ayuda. Sólo el nombre de pila.*

---

### 2. Apellido(s) — *Respuesta corta, obligatoria*

> **Apellido(s)**

*Se separa del nombre para no tener que dividirlo después; la maestra los une
en una sola columna `Nombre`.*

---

### 3. Correo de contacto — *Respuesta corta, obligatoria, validación: dirección de correo*

> **¿A qué correo quieres que te escribamos?**
>
> *Puede ser distinto del correo con que ingresaste a este formulario. Es
> donde recibirás nuestras invitaciones.*

**Esta es la pregunta que hace el trabajo.** Mucha gente tiene cuenta Google
personal e institucional y no revisa con cuál entró; sin esta pregunta, las
invitaciones de trabajo llegan al correo personal de quien no lo quería así.

Validación: *Texto → Dirección de correo electrónico.*

---

### 4. Correo alternativo — *Respuesta corta, opcional, validación: dirección de correo*

> **Correo alternativo (opcional)**
>
> *Por si alguna vez dejas de usar el anterior.*

---

### 5. Institución — *Desplegable con «Otra», opcional*

> **Institución, agrupación o asociación**
>
> *Si no aplica, déjalo en blanco.*

**Debe quedar opcional.** Varias personas de la lista no tienen organización, y
obligar a poner algo sólo genera basura.

Activa **«Otra…»** para que se pueda escribir un valor libre, y revisa esos
valores libres cada cierto tiempo: los que se repitan deben pasar al
desplegable.

Lista inicial abajo (§ Desplegable de instituciones).

---

### 6. Consentimiento — *Selección única, obligatoria*

> **¿Quieres recibir invitaciones a los próximos Encuentros?**
>
> - Sí
> - No

Una sola pregunta, sin párrafos legales. Quien responda **No** igual se
registra en la maestra con `Consentimiento = No`: se conserva el registro de
asistencia y se vuelve imposible escribirle por error.

---

## Bloque de evento (opcional)

Estas preguntas sirven para la logística del evento y **no llegan a la
maestra**. Ponlas al final para que no estorben el bloque de contacto.

- **¿Asistirás?** — Sí / No *(selección única)*
- **¿Requieres algo para participar?** — respuesta corta *(accesibilidad,
  alimentación)*
- **¿Cómo te enteraste de este Encuentro?** — desplegable *(sólo si de verdad
  vas a usar la respuesta)*

---

## Desplegable de instituciones

Derivado de los **113 valores distintos** que hoy tiene la maestra (147
celdas), unificados a **84 organizaciones**. Orden alfabético, que es como se
busca en un desplegable.

Las siglas van con su nombre completo para que la persona reconozca la suya
escrita de cualquiera de las dos formas. **Las marcadas con «?» son expansiones
que deduje y hay que confirmar antes de publicar.**

```
AECID — Agencia Española de Cooperación Internacional para el Desarrollo
Área Tinquilco
Así Conserva Chile
Aula de Mar
Aves Chile
Bollen
CEDEL
CEHUM
Centro Arte Mira
Centro Cultural de España
CEREFAN
CHIC
CIGIDEN
CIMA
CONAF — Corporación Nacional Forestal
Córpora
(CR)² — Centro de Ciencia del Clima y la Resiliencia   (?)
ECIM — Estación Costera de Investigaciones Marinas     (?)
Embajada del Reino Unido en Chile
Explora
FAO Chile
Fondo Naturaleza Chile (FNC)
Fundación Chilco
Fundación Cosmos
Fundación Desierto de Atacama
Fundación Forecos
Fundación Llancalil
Fundación Mallines
Fundación Mar Adentro (FMA)
Fundación Maritorio Resiliente
Fundación Ojos de Mar
Fundación Origen
Fundación Reñihué
Fundación San Carlos de Maipo
Fundación Tierra Austral
Fundación Wayka
Humedal Ojos de Mar
Humedal Tunquén
IEB — Instituto de Ecología y Biodiversidad
Instituto para el Desarrollo Sustentable
Kreen
Laboratorio Natural Andes del Sur
Ladera Sur
Manomet
MIM — Museo Interactivo Mirador                        (?)
Ministerio de Energía
Ministerio de Relaciones Exteriores
Ministerio del Medio Ambiente (MMA)
MSSA — Museo de la Solidaridad Salvador Allende        (?)
Museo Nacional de Historia Natural
Museo Taller
Museo Violeta Parra
National Audubon Society
Observatorio Socioambiental del Aconcagua
Oikonos
ONG Ecosistemas
Packard Foundation
Panthera
Parque Andino Juncal
Patagonia
Pew Trusts
Pingüino Rey
PNUD Chile
Pontificia Universidad Católica de Chile (PUC)
Proyecto Imagina Rural
Reservas Elementales
Robles de Cantillana
ROC — Red de Observadores de Aves y Vida Silvestre     (?)
SBAP — Servicio de Biodiversidad y Áreas Protegidas
SN Alto de Cantillana
SN Cascada de las Ánimas
Sociedad Chilena de Socioecología y Etnoecología (SOSOET)
Symbiotica
Tepual
The Nature Conservancy (TNC)
Universidad Adolfo Ibáñez
Universidad de Chile
Universidad de Concepción
Universidad de los Andes
Universidad Santo Tomás
Wildlife Conservation Society (WCS)
WWF Chile
Independiente
Otra…
```

### Qué se unificó

**Sigla y nombre completo conviviendo en la maestra** — cada par son personas
distintas cuya organización hoy no se puede contar junta:

| Sigla | Nombre completo |
|---|---|
| `FNC` | `Fondo Naturaleza Chile` |
| `MMA` | `Ministerio del Medio Ambiente` |
| `WCS` | `Wildlife Conservation Society` |
| `TNC` | `The Nature Conservancy` |
| `SBAP` | `Servicio de Biodiversidad y Áreas Protegidas` |
| `SOSOET` | `Sociedad Chilena de Socioecología y Etnoecología` |
| `Pew` | `Pew Trusts` |
| `Tierra Austral` | `Fundación Tierra Austral` |
| `FMA` | *(sin nombre completo en la maestra)* |

**Mayúsculas y tildes** — `Fundacion/Fundación Origen`,
`Fundación/fundación Wayka`, `Independiente/independiente`,
`FUNDACION MARITORIO RESILIENTE`.

**Cargo en vez de organización** — 10 celdas guardan el puesto de la persona,
no dónde trabaja. El desplegable se queda con la organización:
`Director MIM` → MIM · `Director WWF Chile` → WWF Chile ·
`Directora MSSA` → MSSA · `Directora Museo Violeta Parra` → Museo Violeta
Parra · `SBAP- Jefa División Biodiversidad`, `SBAP- Jefe División Áreas
Protegidas`, `SBAP- Depto Fondo e IECB` → SBAP ·
`Director Plataforma Cultural U. de Chile` → Universidad de Chile ·
`Patagonia - deportista y escalador` → Patagonia ·
`Coordinador Centros UC` → PUC.

*El cargo es información útil; su lugar es la columna `Notas` de la maestra,
no el nombre de la organización.*

### Tres decisiones que tomé, revisables

1. **Las universidades aparecen una sola vez.** Siete valores son unidades de
   la UC (`Geografía UC`, `CITEC - UC`, `Trabajo Social UC`,
   `Centro Patagonia UC`, `Antropologo UC - Glaciares`,
   `Geografía - UC - Centro del Desierto`, `PUC`) y quedan como
   `Pontificia Universidad Católica de Chile`. Un desplegable sirve para que la
   gente encuentre su institución, no para modelar organigramas; el
   departamento cabe en `Notas`. **Si prefieres que centros con identidad
   propia —`CITEC`, `Centro Patagonia UC`— aparezcan aparte, dímelo.**

2. **Los socios de Así Conserva Chile van como organizaciones propias.**
   `Fundación San Carlos de Maipo`, `Reservas Elementales`,
   `Robles de Cantillana`, `SN Cascada de las Ánimas` y `SN Alto de Cantillana`
   son entidades reales; «Socios de Así Conserva Chile» describe una relación,
   no un empleador. La red también está en la lista.

3. **Lo que no es una organización quedó fuera:** `Ciudadano preocupado`,
   `Curadora de arte`, `Artista independiente`, `Ciencia ambiental`. Quien se
   describa así elige `Independiente` o escribe en `Otra…`.

**Dobles afiliaciones** (`FAO Chile, Universidad de Chile`,
`CEDEL - CIGIDEN`): las dos partes están en la lista por separado. El
desplegable admite una sola respuesta; la segunda organización va en `Notas`.

*Unificar los valores que ya están escritos en la maestra es otra tarea —
tocaría filas que hoy están como su dueño las dejó, y no se ha hecho.*

---

## Correspondencia con la lista maestra

| Formulario | → | Maestra |
|---|---|---|
| P1 + P2 unidas | → | `Nombre` |
| P5 | → | `Organización` |
| P3 | → | `Email principal` |
| P4, o el correo verificado si difiere | → | `Email alternativo` |
| P6 | → | `Consentimiento` |
| *(bandera `--origen`)* | → | `Origen` |
| *(bandera `--fecha`)* | → | `Fecha` |
| *(lo escribe el equipo)* | → | `Notas` |
| *(lo calcula el script)* | → | `N` |

`Origen` y `Fecha` **no se preguntan.** Una pregunta que la gente responde es
una pregunta que la gente responde mal; ambas se pasan al ejecutar el merge:

```bash
python scripts/merge_contacts.py --apply \
    --master "LISTADO_CONTACTOS_MAESTRO.xlsx" \
    --origen "4° Encuentro Hablemos de Conservación" \
    --fecha 2026-11-15
```

Las columnas del bloque de evento (`¿Asistirás?`, marca temporal, puntuación)
son ignoradas por el script.

---

## Antes de publicar cada formulario

1. Duplicar la plantilla, no crear uno nuevo.
2. Actualizar título y descripción con el nombre del Encuentro.
3. Revisar el desplegable: agregar las organizaciones que hayan aparecido como
   «Otra» desde el evento anterior.
4. Confirmar que **Recopilar direcciones → Verificada** sigue activo (se pierde
   al duplicar en algunas versiones de Forms).
5. Responderlo tú mismo una vez y correr `--review` sobre esa única respuesta.
   Si sale con `confianza = alta` y sin observaciones, el formulario está bien
   armado.
