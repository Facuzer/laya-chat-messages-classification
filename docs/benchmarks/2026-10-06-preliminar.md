# Scoreboard de moderación

> **PRELIMINAR:** incluye filas de la suite que nadie revisó todavía. Sirve para probar la herramienta, no para decidir.

Generado: 2026-10-06T19:49:26Z · Remuestreos bootstrap: 1000

## Cómo leer esto

- **AUROC [IC 95%]**: probabilidad de que el modelo puntúe más alto un mensaje dañino que uno sano. 0.50 es azar y 1.00 es perfecto. El intervalo muestra cuánto puede variar por azar.
- **R · FPR**: con el umbral de operación de cada modelo (el que marca al 5% de los mensajes sanos de la partición de calibración), qué parte de los dañinos detecta (R) y qué parte de los sanos marca (FPR) en esa fuente.
- **pocos datos (a+/b−)**: menos de 30 ejemplos de alguna clase; no sacar conclusiones.
- **n/a**: el modelo no responde esa categoría. **—**: la fuente no tiene etiquetas de esa categoría.
- Estas fuentes tienen muchos más mensajes dañinos que un chat real (donde son el 1–5%): en producción la precisión va a ser menor.

## Cobertura (ejemplos de evaluación: dañinos/sanos)

| fuente | idioma | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|---|
| hatecheck-en | en | 52/476 | 217/420 | 1995/906 | — | 168/0 | 2047/420 | 52/0 |
| hatecheck-es | es | 43/520 | 229/466 | 2056/834 | — | 196/0 | 2099/466 | 43/0 |
| hatecheck-pt | pt | 37/516 | 231/466 | 2052/765 | — | 193/0 | 2089/466 | 37/0 |
| olidbr | pt | 896/264 | 0/229 | 198/1153 | — | 453/917 | 948/229 | 462/356 |
| suite-es | es | 138/271 | 42/271 | 48/280 | 19/65 | 44/65 | 228/271 | 203/6 |
| suite-es/generated | es | 138/271 | 42/271 | 48/280 | 19/65 | 44/65 | 228/271 | 203/6 |
| toldbr | pt | 286/2566 | 0/2500 | 63/3069 | — | 350/2248 | 326/2500 | — |

## ¿Hay que moderar? (`flag` = insult, threat o identity_hate)

| modelo | hatecheck-en | hatecheck-es | hatecheck-pt | olidbr | suite-es | suite-es/generated | toldbr |
|---|---|---|---|---|---|---|---|
| laya | 0.89 [0.85–0.93]<br>R 25% · FPR 1% | 0.79 [0.75–0.84]<br>R 32% · FPR 8% | 0.78 [0.74–0.82]<br>R 37% · FPR 8% | 0.64 [0.61–0.68]<br>R 11% · FPR 4% | 0.75 [0.71–0.79]<br>R 11% · FPR 1% | 0.75 [0.71–0.79]<br>R 11% · FPR 1% | 0.72 [0.69–0.75]<br>R 16% · FPR 5% |
| laya-app | 0.86 [0.82–0.89]<br>R 7% · FPR 0% | 0.69 [0.64–0.73]<br>R 12% · FPR 5% | 0.69 [0.65–0.73]<br>R 17% · FPR 5% | 0.69 [0.65–0.72]<br>R 13% · FPR 3% | 0.65 [0.60–0.70]<br>R 6% · FPR 1% | 0.65 [0.60–0.70]<br>R 6% · FPR 1% | 0.73 [0.69–0.75]<br>R 23% · FPR 5% |
| detoxify | 0.83 [0.78–0.87]<br>R 66% · FPR 16% | 0.80 [0.75–0.84]<br>R 49% · FPR 13% | 0.81 [0.77–0.84]<br>R 49% · FPR 11% | 0.68 [0.64–0.72]<br>R 12% · FPR 4% | 0.75 [0.71–0.80]<br>R 9% · FPR 1% | 0.75 [0.71–0.80]<br>R 9% · FPR 1% | 0.80 [0.78–0.83]<br>R 7% · FPR 1% |
| horizon-mmbert | 0.75 [0.71–0.80]<br>R 49% · FPR 14% | 0.76 [0.72–0.80]<br>R 49% · FPR 14% | 0.76 [0.71–0.80]<br>R 43% · FPR 11% | 0.73 [0.70–0.76]<br>R 12% · FPR 1% | 0.75 [0.71–0.79]<br>R 24% · FPR 5% | 0.75 [0.71–0.79]<br>R 24% · FPR 5% | 0.80 [0.78–0.83]<br>R 17% · FPR 1% |

## Diferencia de AUROC contra el mejor de cada fuente (`flag`, bootstrap pareado)

Si el intervalo incluye 0, no hay evidencia de que el mejor sea mejor de verdad.

| fuente | mejor | modelo | diferencia [IC 95%] |
|---|---|---|---|
| hatecheck-en | laya | laya-app | -0.03 [-0.07 – 0.01] |
| hatecheck-en | laya | detoxify | -0.06 [-0.11 – -0.02] |
| hatecheck-en | laya | horizon-mmbert | -0.14 [-0.18 – -0.10] |
| hatecheck-es | detoxify | laya | -0.00 [-0.04 – 0.04] |
| hatecheck-es | detoxify | laya-app | -0.11 [-0.15 – -0.07] |
| hatecheck-es | detoxify | horizon-mmbert | -0.04 [-0.07 – -0.00] |
| hatecheck-pt | detoxify | laya | -0.03 [-0.06 – 0.01] |
| hatecheck-pt | detoxify | laya-app | -0.11 [-0.16 – -0.07] |
| hatecheck-pt | detoxify | horizon-mmbert | -0.05 [-0.08 – -0.02] |
| olidbr | horizon-mmbert | laya | -0.09 [-0.12 – -0.05] |
| olidbr | horizon-mmbert | laya-app | -0.04 [-0.08 – -0.01] |
| olidbr | horizon-mmbert | detoxify | -0.05 [-0.08 – -0.01] |
| suite-es | detoxify | laya | -0.00 [-0.05 – 0.04] |
| suite-es | detoxify | laya-app | -0.11 [-0.15 – -0.06] |
| suite-es | detoxify | horizon-mmbert | -0.00 [-0.03 – 0.03] |
| suite-es/generated | detoxify | laya | -0.00 [-0.05 – 0.04] |
| suite-es/generated | detoxify | laya-app | -0.11 [-0.15 – -0.06] |
| suite-es/generated | detoxify | horizon-mmbert | -0.00 [-0.03 – 0.03] |
| toldbr | horizon-mmbert | laya | -0.09 [-0.11 – -0.06] |
| toldbr | horizon-mmbert | laya-app | -0.08 [-0.10 – -0.05] |
| toldbr | horizon-mmbert | detoxify | -0.00 [-0.02 – 0.02] |

## Por categoría (AUROC)

### hatecheck-en (en)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.73 [0.63–0.80] | 0.99 [0.98–1.00] | 0.84 [0.80–0.88] | — | pocos datos (168+/0−) | 0.89 [0.85–0.93] | pocos datos (52+/0−) |
| laya-app | n/a | n/a | n/a | — | n/a | 0.86 [0.82–0.89] | n/a |
| detoxify | 0.76 [0.68–0.83] | 0.98 [0.95–1.00] | 0.70 [0.67–0.74] | — | pocos datos (168+/0−) | 0.83 [0.78–0.87] | n/a |
| horizon-mmbert | 0.66 [0.55–0.75] | 0.99 [0.96–1.00] | 0.74 [0.70–0.78] | — | pocos datos (168+/0−) | 0.75 [0.71–0.80] | n/a |

### hatecheck-es (es)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.54 [0.44–0.63] | 0.97 [0.94–0.99] | 0.70 [0.66–0.75] | — | pocos datos (196+/0−) | 0.79 [0.75–0.84] | pocos datos (43+/0−) |
| laya-app | n/a | n/a | n/a | — | n/a | 0.69 [0.64–0.73] | n/a |
| detoxify | 0.77 [0.67–0.84] | 0.98 [0.95–0.99] | 0.71 [0.67–0.74] | — | pocos datos (196+/0−) | 0.80 [0.75–0.84] | n/a |
| horizon-mmbert | 0.66 [0.55–0.77] | 0.99 [0.97–1.00] | 0.72 [0.68–0.76] | — | pocos datos (196+/0−) | 0.76 [0.72–0.80] | n/a |

### hatecheck-pt (pt)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.56 [0.46–0.66] | 0.96 [0.93–0.99] | 0.72 [0.68–0.75] | — | pocos datos (193+/0−) | 0.78 [0.74–0.82] | pocos datos (37+/0−) |
| laya-app | n/a | n/a | n/a | — | n/a | 0.69 [0.65–0.73] | n/a |
| detoxify | 0.71 [0.63–0.80] | 0.98 [0.95–1.00] | 0.70 [0.67–0.74] | — | pocos datos (193+/0−) | 0.81 [0.77–0.84] | n/a |
| horizon-mmbert | 0.66 [0.54–0.77] | 0.98 [0.96–1.00] | 0.72 [0.68–0.76] | — | pocos datos (193+/0−) | 0.76 [0.71–0.80] | n/a |

### olidbr (pt)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.67 [0.63–0.70] | pocos datos (0+/229−) | 0.62 [0.58–0.66] | — | 0.56 [0.53–0.59] | 0.64 [0.61–0.68] | 0.57 [0.53–0.61] |
| laya-app | n/a | n/a | n/a | — | n/a | 0.69 [0.65–0.72] | n/a |
| detoxify | 0.70 [0.66–0.73] | pocos datos (0+/229−) | 0.72 [0.68–0.75] | — | 0.71 [0.68–0.74] | 0.68 [0.64–0.72] | n/a |
| horizon-mmbert | 0.74 [0.71–0.77] | pocos datos (0+/229−) | 0.69 [0.65–0.73] | — | 0.76 [0.73–0.79] | 0.73 [0.70–0.76] | n/a |

### suite-es (es)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.72 [0.67–0.77] | 0.73 [0.64–0.82] | 0.80 [0.72–0.87] | pocos datos (19+/65−) | 0.84 [0.75–0.91] | 0.75 [0.71–0.79] | pocos datos (203+/6−) |
| laya-app | n/a | n/a | n/a | n/a | n/a | 0.65 [0.60–0.70] | n/a |
| detoxify | 0.82 [0.78–0.86] | 0.78 [0.68–0.86] | 0.76 [0.69–0.83] | pocos datos (19+/65−) | 0.93 [0.87–0.98] | 0.75 [0.71–0.80] | n/a |
| horizon-mmbert | 0.82 [0.78–0.86] | 0.79 [0.69–0.87] | 0.78 [0.70–0.85] | pocos datos (19+/65−) | 0.98 [0.95–1.00] | 0.75 [0.71–0.79] | n/a |

### suite-es/generated (es)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.72 [0.67–0.77] | 0.73 [0.64–0.82] | 0.80 [0.72–0.87] | pocos datos (19+/65−) | 0.84 [0.75–0.91] | 0.75 [0.71–0.79] | pocos datos (203+/6−) |
| laya-app | n/a | n/a | n/a | n/a | n/a | 0.65 [0.60–0.70] | n/a |
| detoxify | 0.82 [0.78–0.86] | 0.78 [0.68–0.86] | 0.76 [0.69–0.83] | pocos datos (19+/65−) | 0.93 [0.87–0.98] | 0.75 [0.71–0.80] | n/a |
| horizon-mmbert | 0.82 [0.78–0.86] | 0.79 [0.69–0.87] | 0.78 [0.70–0.85] | pocos datos (19+/65−) | 0.98 [0.95–1.00] | 0.75 [0.71–0.79] | n/a |

### toldbr (pt)

| modelo | insult | threat | identity_hate | sexual_harassment | profanity | flag | targets_player |
|---|---|---|---|---|---|---|---|
| laya | 0.74 [0.71–0.77] | pocos datos (0+/2500−) | 0.59 [0.52–0.66] | — | 0.68 [0.65–0.70] | 0.72 [0.69–0.75] | — |
| laya-app | n/a | n/a | n/a | — | n/a | 0.73 [0.69–0.75] | — |
| detoxify | 0.83 [0.80–0.85] | pocos datos (0+/2500−) | 0.63 [0.55–0.70] | — | 0.78 [0.76–0.81] | 0.80 [0.78–0.83] | — |
| horizon-mmbert | 0.83 [0.81–0.85] | pocos datos (0+/2500−) | 0.65 [0.58–0.71] | — | 0.80 [0.77–0.82] | 0.80 [0.78–0.83] | — |

## Por funcionalidad (aciertos con el umbral de operación)

En HateCheck se mide `identity_hate`, que es lo que HateCheck evalúa; en la suite propia, `flag`.

### hatecheck-en

| funcionalidad | n | laya | laya-app | detoxify | horizon-mmbert |
|---|---|---|---|---|---|
| counter_quote_nh | 131 | 99% | n/a | 63% | 79% |
| counter_ref_nh | 106 | 100% | n/a | 73% | 82% |
| derog_dehum_h | 119 | 13% | n/a | 64% | 63% |
| derog_impl_h | 126 | 4% | n/a | 29% | 29% |
| derog_neg_attrib_h | 98 | 6% | n/a | 58% | 59% |
| derog_neg_emote_h | 105 | 6% | n/a | 38% | 30% |
| ident_neutral_nh | 63 | 100% | n/a | 97% | 100% |
| ident_pos_nh | 161 | 99% | n/a | 93% | 89% |
| negate_neg_nh | 112 | 100% | n/a | 86% | 72% |
| negate_pos_h | 84 | 0% | n/a | 38% | 29% |
| phrase_opinion_h | 105 | 1% | n/a | 55% | 52% |
| phrase_question_h | 112 | 0% | n/a | 50% | 50% |
| profanity_h | 84 | 7% | n/a | 62% | 63% |
| profanity_nh | 84 | 100% | n/a | 100% | 100% |
| ref_subs_clause_h | 105 | 4% | n/a | 48% | 35% |
| ref_subs_sent_h | 119 | 1% | n/a | 35% | 33% |
| slur_h | 108 | 11% | n/a | 10% | 1% |
| slur_homonym_nh | 22 | 100% | n/a | 100% | 95% |
| slur_reclaimed_nh | 67 | 100% | n/a | 72% | 100% |
| spell_char_del_h | 112 | 4% | n/a | 27% | 34% |
| spell_char_swap_h | 84 | 1% | n/a | 40% | 29% |
| spell_leet_h | 138 | 11% | n/a | 23% | 17% |
| spell_space_add_h | 152 | 1% | n/a | 19% | 14% |
| spell_space_del_h | 127 | 13% | n/a | 35% | 31% |
| target_group_nh | 52 | 100% | n/a | 100% | 100% |
| target_indiv_nh | 52 | 96% | n/a | 100% | 100% |
| target_obj_nh | 56 | 100% | n/a | 100% | 100% |
| threat_dir_h | 112 | 12% | n/a | 59% | 56% |
| threat_norm_h | 105 | 4% | n/a | 53% | 56% |

### hatecheck-es

| funcionalidad | n | laya | laya-app | detoxify | horizon-mmbert |
|---|---|---|---|---|---|
| counter_quote_nh | 133 | 83% | n/a | 77% | 77% |
| counter_ref_nh | 104 | 85% | n/a | 94% | 90% |
| derog_dehum_h | 105 | 17% | n/a | 54% | 60% |
| derog_impl_h | 91 | 10% | n/a | 30% | 25% |
| derog_neg_attrib_h | 112 | 19% | n/a | 50% | 56% |
| derog_neg_emote_h | 111 | 12% | n/a | 28% | 40% |
| ident_neutral_nh | 119 | 96% | n/a | 100% | 99% |
| ident_pos_nh | 159 | 97% | n/a | 92% | 82% |
| negate_neg_nh | 111 | 96% | n/a | 79% | 67% |
| negate_pos_h | 124 | 2% | n/a | 25% | 19% |
| phrase_opinion_h | 110 | 25% | n/a | 39% | 58% |
| phrase_question_h | 126 | 10% | n/a | 25% | 42% |
| profanity_h | 119 | 23% | n/a | 26% | 55% |
| profanity_nh | 77 | 96% | n/a | 100% | 100% |
| ref_subs_clause_h | 98 | 20% | n/a | 28% | 38% |
| ref_subs_sent_h | 124 | 24% | n/a | 21% | 34% |
| slur_h | 140 | 29% | n/a | 1% | 0% |
| spell_char_del_h | 105 | 16% | n/a | 13% | 21% |
| spell_char_swap_h | 97 | 25% | n/a | 28% | 39% |
| spell_leet_h | 160 | 19% | n/a | 10% | 26% |
| spell_space_add_h | 99 | 5% | n/a | 12% | 22% |
| spell_space_del_h | 106 | 29% | n/a | 32% | 49% |
| target_group_nh | 34 | 100% | n/a | 100% | 100% |
| target_indiv_nh | 43 | 95% | n/a | 100% | 100% |
| target_obj_nh | 54 | 98% | n/a | 100% | 100% |
| threat_dir_h | 119 | 39% | n/a | 30% | 55% |
| threat_norm_h | 110 | 44% | n/a | 55% | 65% |

### hatecheck-pt

| funcionalidad | n | laya | laya-app | detoxify | horizon-mmbert |
|---|---|---|---|---|---|
| counter_quote_nh | 92 | 83% | n/a | 76% | 85% |
| counter_ref_nh | 91 | 91% | n/a | 89% | 92% |
| derog_dehum_h | 133 | 32% | n/a | 50% | 49% |
| derog_impl_h | 124 | 11% | n/a | 15% | 16% |
| derog_neg_attrib_h | 126 | 30% | n/a | 52% | 48% |
| derog_neg_emote_h | 112 | 12% | n/a | 25% | 29% |
| ident_neutral_nh | 108 | 94% | n/a | 100% | 100% |
| ident_pos_nh | 167 | 95% | n/a | 95% | 86% |
| negate_neg_nh | 110 | 94% | n/a | 86% | 79% |
| negate_pos_h | 91 | 9% | n/a | 31% | 16% |
| phrase_opinion_h | 126 | 38% | n/a | 33% | 52% |
| phrase_question_h | 105 | 18% | n/a | 25% | 33% |
| profanity_h | 112 | 38% | n/a | 32% | 51% |
| profanity_nh | 81 | 96% | n/a | 100% | 100% |
| ref_subs_clause_h | 133 | 24% | n/a | 29% | 32% |
| ref_subs_sent_h | 125 | 35% | n/a | 22% | 30% |
| slur_h | 93 | 11% | n/a | 0% | 0% |
| spell_char_del_h | 112 | 19% | n/a | 15% | 18% |
| spell_char_swap_h | 119 | 27% | n/a | 26% | 40% |
| spell_leet_h | 103 | 27% | n/a | 8% | 17% |
| spell_space_add_h | 94 | 9% | n/a | 5% | 17% |
| spell_space_del_h | 113 | 27% | n/a | 21% | 40% |
| target_group_nh | 29 | 90% | n/a | 100% | 100% |
| target_indiv_nh | 37 | 89% | n/a | 100% | 100% |
| target_obj_nh | 50 | 96% | n/a | 100% | 100% |
| threat_dir_h | 119 | 39% | n/a | 35% | 57% |
| threat_norm_h | 112 | 32% | n/a | 48% | 53% |

### suite-es

| funcionalidad | n | laya | laya-app | detoxify | horizon-mmbert |
|---|---|---|---|---|---|
| affectionate_identity_term | 7 | 100% | 100% | 100% | 100% |
| banter_friendly | 35 | 97% | 100% | 100% | 100% |
| bets_amounts | 23 | 100% | 100% | 100% | 96% |
| code_mixing_banter | 8 | 100% | 100% | 100% | 75% |
| code_mixing_insult | 10 | 20% | 30% | 20% | 30% |
| criticism_no_insult | 28 | 100% | 100% | 100% | 100% |
| everyday | 42 | 100% | 100% | 100% | 100% |
| gaming_insult | 23 | 4% | 0% | 13% | 22% |
| identity_hate | 19 | 16% | 16% | 21% | 37% |
| identity_hate_casino | 16 | 19% | 0% | 0% | 0% |
| insult_direct | 82 | 10% | 7% | 12% | 41% |
| negation | 26 | 96% | 100% | 92% | 69% |
| obfuscated_insult | 23 | 0% | 0% | 0% | 0% |
| obfuscated_slur | 13 | 0% | 0% | 0% | 0% |
| profanity_at_game | 12 | 100% | 100% | 100% | 100% |
| profanity_untargeted | 32 | 100% | 94% | 100% | 100% |
| quoting_reporting | 16 | 94% | 94% | 88% | 88% |
| quoting_reporting_casino | 9 | 100% | 100% | 100% | 100% |
| sarcasm | 25 | 96% | 96% | 100% | 100% |
| self_deprecation | 8 | 100% | 100% | 100% | 100% |
| threat | 31 | 23% | 0% | 3% | 13% |
| threat_doxxing | 11 | 0% | 9% | 0% | 9% |

## Laya: mensajes que su router mandó al checkpoint inglés

| modelo | fuente | al inglés |
|---|---|---|
| laya | hatecheck-en | 92% |
| laya | hatecheck-es | 0% |
| laya | hatecheck-pt | 1% |
| laya | olidbr | 1% |
| laya | suite-es | 2% |
| laya | toldbr | 1% |
| laya-app | hatecheck-en | 92% |
| laya-app | hatecheck-es | 0% |
| laya-app | hatecheck-pt | 1% |
| laya-app | olidbr | 1% |
| laya-app | suite-es | 2% |
| laya-app | toldbr | 1% |

## Latencia

| modelo | dispositivo | idioma | p50 ms | p95 ms | mensajes/s en lotes |
|---|---|---|---|---|---|
| laya | cpu | es | 404 | 527 | 3 |
| laya-app | cpu | es | 124 | 159 | 9 |
| detoxify | cpu | es | 18 | 28 | 94 |
| horizon-mmbert | cpu | es | 28 | 42 | 77 |
| laya | cuda | es | 50 | 65 | 3 |
| laya-app | cuda | es | 38 | 51 | 38 |
| detoxify | cuda | es | 9 | 14 | 383 |
| horizon-mmbert | cuda | es | 18 | 21 | 279 |

## Modelos

- **laya** (`qa5d7896f-laya0.3.22-default-multilingual`, Apache-2.0): Laya sin ajustar (zero-shot) con seis preguntas redactadas para este benchmark y sin pulir: el resultado depende tanto de la redacción como del modelo. El inglés que Laya identifica va al checkpoint inglés (ModernBERT-large) y el resto al multilingüe (mmBERT-base). Hace una fila de encoder por pregunta: seis por mensaje. Umbrales de operación: insult=0.878, threat=0.576, identity_hate=0.953, sexual_harassment=0.256, profanity=0.997, flag=0.949, targets_player=0.974.
- **laya-app** (`qcabfaa77-laya0.3.22-default-multilingual`, Apache-2.0): La pregunta `insult` que usa hoy la app (insultos, slurs y amenazas en una sola pregunta) como flag. Es la línea de base: lo que ya tenemos. Umbrales de operación: flag=0.864.
- **detoxify** (`multilingual-0.5.2`, Apache-2.0): XLM-R entrenado con Jigsaw 2020. Fuera de `toxicity`, sus categorías se aprendieron de etiquetas en inglés traducidas, así que en español y portugués son débiles. `sexual_explicit` detecta contenido sexual, no acoso. Umbrales de operación: insult=0.581, threat=0.0193, identity_hate=0.683, sexual_harassment=0.0173, profanity=0.312, flag=0.645.
- **horizon-mmbert** (`dbf12a991527`, Apache-2.0): mmBERT-base ajustado con Civil Comments traducido por un LLM. Lo publicó una cuenta creada en septiembre de 2026, sin trayectoria, y no sabemos con qué datos exactos se entrenó: si vio ToLD-Br, OLID-BR o HateCheck, sus números en esas fuentes están inflados. Se carga solo en safetensors y sin código remoto. Umbrales de operación: insult=0.507, threat=0.116, identity_hate=0.558, sexual_harassment=0.0693, profanity=0.439, flag=0.576.

## Fuentes

| fuente | idioma | licencia | ejemplos |
|---|---|---|---|
| suite-es | es | propia | 651 |
| hatecheck-es | es | CC BY 4.0 | 3655 |
| hatecheck-pt | pt | CC BY 4.0 | 3539 |
| hatecheck-en | en | CC BY 4.0 | 3728 |
| toldbr | pt | CC BY-SA 4.0 | 4000 |
| olidbr | pt | CC BY 4.0 | 1738 |
