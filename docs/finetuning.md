# Enseñarle a Laya: fine-tuning

## La idea en cuatro frases

Laya ya "entiende" texto, pero no conoce el rioplatense ni tu criterio de qué es un insulto. El fine-tuning le muestra cientos de ejemplos con la respuesta correcta y ajusta un poco sus pesos (sus "neuronas") para que acierte más en ese tipo de mensajes. Después se lo evalúa con ejemplos que **nunca vio** durante el entrenamiento, para saber si de verdad mejoró y no solo memorizó. Solo se usa el modelo nuevo si le gana al original.

## El circuito

1. **Juntar ejemplos etiquetados**, por tres vías que terminan en el mismo dataset (`data/training/examples.json`):
   - **Chat:** debajo de cada mensaje, "¿Es insulto? Sí / No".
   - **Pantalla de entrenamiento** (`/training.html`): muestra de a uno los ejemplos pendientes, con la sugerencia y lo que opina Laya hoy. Atajos: `1` insulto, `2` no insulto, `3` descartar.
   - **Importar un archivo:** `uv run python scripts/import_examples.py archivo.csv` (columnas `text` y, opcionalmente, `label` con `insult` o `clean`).
2. **Entrenar:** `uv run python scripts/finetune.py`. En tu 3080 Ti tarda menos de un minuto con ~400 ejemplos.
3. **Leer el reporte.** Si el modelo nuevo es mejor, queda activado. Reiniciá el server para usarlo.

Los 421 ejemplos de `datasets/generated_rioplatense_v1.jsonl` ya están importados como **pendientes**: cada uno trae una etiqueta sugerida, pero cuenta recién cuando lo revisás.

## Criterio de etiquetado

- **Insulto:** insultos ("sos un pelotudo"), slurs ("mogólico"), amenazas ("te voy a romper la cara") y groserías fuertes, aunque no apunten a nadie.
- **No insulto:** todo lo demás. Sarcasmo ("qué genio que sos"), malas palabras suaves sin destinatario ("qué día de mierda", "carajo"), trato amistoso ("boludo, vení"), negaciones ("no sos un idiota") y hablar sobre insultos ("me dijo forro y me dolió").

Ser consistente importa más que la cantidad: el modelo aprende exactamente el criterio que reflejan tus etiquetas. Ante la duda, "Descartar".

## Cómo leer el reporte

| Línea | Qué significa |
|---|---|
| Aciertos | de todos los mensajes de prueba, cuántos clasificó bien |
| Insultos detectados (recall) | de los insultos reales, cuántos tapó |
| Alarmas correctas (precision) | de los que tapó, cuántos eran insultos de verdad |
| F1 | un solo número que balancea las dos anteriores; es el que decide |
| Error de calibración | cuánto se aleja el "84% seguro" de la realidad; menor es mejor |
| Sentimiento igual al original | el entrenamiento es solo de insultos; esto vigila que no se haya roto el sentimiento |

El modelo nuevo se activa solo si:

- el F1 mejora;
- el sentimiento coincide con el original en el 85% de los mensajes o más;
- hay al menos 10 ejemplos de prueba de cada tipo.

Los mensajes que cambiaron de sentimiento aparecen listados, para que veas si el cambio tiene sentido.

## Por qué hay tres grupos de ejemplos

Cada ejemplo cae, según su texto, en uno de tres grupos fijos:

- **entrenamiento (70%):** de acá aprende;
- **calibración (10%):** se usa para ajustar que sus porcentajes sean honestos;
- **prueba (20%):** el examen final, que nunca ve mientras aprende.

El mismo texto siempre cae en el mismo grupo, así una frase repetida nunca está a la vez en entrenamiento y en el examen.

## Cuántos ejemplos hacen falta

- **Mínimo para entrenar:** 20 insultos y 20 no insultos en el grupo de entrenamiento.
- **Recomendado:** unos 800 en total, con muchos casos difíciles (negaciones, "boludo" amistoso, sarcasmo, amenazas).

Con pocos ejemplos el resultado es poco confiable aunque el reporte diga que mejoró.

## Volver atrás

```
uv run python scripts/activate_model.py --list    # modelos entrenados (* = activo)
uv run python scripts/activate_model.py --base    # volver al modelo original de Laya
uv run python scripts/activate_model.py models/ft-20261001-120000
```

Cada entrenamiento se guarda en su propia carpeta dentro de `models/`. Ninguno se pisa.

## Qué pasa por dentro

`trainer/finetune.py` sigue la receta del notebook oficial de Laya, adaptada a una sola GPU:

- **Entrenamiento:** cross-entropy más el término RLCD, que recompensa probabilidades honestas con una regla de puntuación propia.
- **Sentimiento:** cada texto de entrenamiento también lleva la pregunta de sentimiento, con la respuesta del modelo original como objetivo (destilación), así no se olvida de cómo lo hacía.
- **Calibración:** después se ajustan las temperaturas con el grupo de calibración.
- **Mismo formato que en producción:** las secuencias se arman con el mismo código que usa `predict`, así el modelo entrena con exactamente el formato que ve en producción.
- **Versión fija:** esas funciones son internas de Laya, por eso `pyproject.toml` fija `laya==0.3.22`.

El checkpoint guarda en `training_meta.json` las preguntas con las que se entrenó. Si cambiás la redacción de `QUESTIONS` en `app/classifier.py`, el server avisa al arrancar: hay que volver a entrenar.
