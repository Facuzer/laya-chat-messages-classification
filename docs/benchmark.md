# Benchmark de moderación

## Para qué sirve

Antes de elegir un modelo de moderación los evaluamos a todos con el mismo examen: mensajes en español, portugués e inglés con la respuesta correcta ya etiquetada. El resultado (`bench_out/scoreboard.md`) muestra qué tan bien separa cada modelo los mensajes que hay que moderar de los que no, por fuente y por categoría.

La parte más importante del examen es la **suite propia** (`suites/es_casino.csv`): mensajes como los que vamos a ver en el chat del casino, con lunfardo, jerga de póker y trampas. Una fila cuenta recién cuando una persona revisó sus etiquetas.

## Categorías

| Columna | Es 1 cuando… | Ejemplos |
|---|---|---|
| `insult` | insulta, ridiculiza o acosa a una persona | "sos un manco", "callate gil", "p3lotud0" |
| `threat` | amenaza con daño o violencia, o con escrachar | "te voy a romper la cara", "sé dónde vivís" |
| `identity_hate` | ataca a gente por un rasgo protegido (lista abajo) | "bolita de mierda", "andá a lavar los platos" |
| `sexual_harassment` | comentarios, pedidos o avances sexuales no deseados hacia una persona | "mandame fotos", "pasame tu número, linda" |
| `profanity` | tiene puteadas o lenguaje vulgar, dirigido o no | "qué mano de mierda", "la puta madre" |

**Rasgos protegidos** (`identity_hate`): origen étnico o color de piel, nacionalidad, religión, género, orientación sexual, identidad de género y discapacidad. Los insultos por clase social ("villero", "planero"), por el cuerpo ("gorda", "enano") o por la edad ("viejo choto") son `insult`, no `identity_hate`.

**`target`**: a quién ataca el mensaje, si es dañino.
- `player`: a una persona concreta ("sos un manco").
- `group`: a un grupo ("los bolivianos son…").
- `none`: a nadie, o el mensaje no es dañino.

**Valores:** `1` sí, `0` no, `?` no estamos seguros o no nos ponemos de acuerdo (no se evalúa), vacío = sin revisar.

### Casos difíciles

- **Trato amistoso:** "boludo, vení", "negro, arrancamos", "gordo, buena mano" van con todo en `0`. El mismo término usado con odio ("negro de mierda") es `identity_hate=1`.
- **Contar un insulto no es insultar:** "me dijo trolo, ya lo reporté" va en `0`.
- **Negaciones:** "no sos un idiota" va en `0`.
- **Puteadas sin destinatario:** "qué día de mierda" lleva `profanity=1` y el resto en `0`.
- **Autocrítica:** "soy un manco" va en `0`.
- **Plata y apuestas:** "all in 500", "pago x2", "gg wp" van con todo en `0`. Ojo con los números que parecen leetspeak.
- **Lo ofuscado cuenta igual:** "p e l o t u d o" o "tr0l0" se etiquetan como si estuvieran bien escritos.
- **Mezcla de idiomas:** "nice hand manco" es `insult=1`.
- **Sarcasmo:** "qué genio que sos" va en `0`, salvo que ridiculice a alguien de forma clara.

## Cómo revisar la suite

1. Abrí `suites/es_casino.csv` en Excel o Google Sheets. Está separado por `;`.
2. En cada fila, leé el texto, corregí las categorías y `target`, y poné `reviewed` en `1`. Las sugerencias vienen de un generador: no las des por buenas.
3. Si una fila no tiene sentido, borrala.
4. **Sumá mensajes reales de memoria:** al menos 50 frases que hayas visto en chats de juegos o de póker, con `author` en `human`. Son las más valiosas, y el reporte las separa de las generadas.
5. Guardá como CSV (UTF-8). Sirve tanto con `;` como con `,`. Ojo: Excel convierte en fórmula un texto que empieza con `=`, `+` o `-`; si te pasa, poné un espacio adelante.
6. **Segunda opinión:** que otra persona revise una copia de las mismas ~150 filas sin mirar la primera revisión, y compará las dos:
   `uv run python scripts/bench.py agreement suites/es_casino.csv copia.csv`
   Si el kappa de una categoría da menos de 0.6, el criterio no está claro: hablenlo y ajusten esta guía.
