# Guia de anotacion Tarea 2

## Objetivo

Validar si el fragmento declara una contribucion cientifica propia del
trabajo. La etiqueta heuristica es solo una ayuda: el anotador puede corregirla.

## Etiquetas manuales

Usar exactamente una de estas dos etiquetas:

- `contribucion`
- `no_contribucion`

## Marcar como contribucion

Marcar `contribucion` cuando el fragmento diga que el trabajo propio propone,
presenta, desarrolla, introduce, aporta, construye o valida algo propio:

- metodo, metodologia, modelo, sistema, herramienta o algoritmo
- enfoque, marco, arquitectura, estrategia o protocolo
- evidencia, hallazgo, corpus, dataset o recurso propio
- contribucion principal, aporte o solucion del articulo

La contribucion debe estar atribuida al trabajo, articulo, estudio,
investigacion o autores del fragmento.

## Marcar como no_contribucion

Marcar `no_contribucion` cuando el fragmento:

- describe contexto, antecedentes, marco teorico, metodologia o resultados sin
  declarar aporte propio
- menciona contribuciones de otros autores
- solo declara objetivos del estudio sin novedad o aporte explicito
- resume trabajos previos
- contiene senales como "propone" o "aporte", pero referidas a terceros
- esta demasiado degradado por OCR como para validar una contribucion

## Campos a llenar

En `tarea2_muestra_anotacion_10pct.csv`, llenar:

- `etiqueta_manual`: `contribucion` o `no_contribucion`
- `confianza_anotador`: alta, media o baja
- `observaciones_anotador`: opcional, explicar dudas o falsos positivos
- `anotador`: nombre o iniciales

## Regla practica

Si el fragmento no permite responder "que aporta este trabajo?" con evidencia
textual clara, marcar `no_contribucion`.
