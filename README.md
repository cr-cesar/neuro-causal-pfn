# Neuro-Causal-PFN

Modelo fundacional causal basado en neuroimagen para la estimacion de efectos
individualizados del tratamiento en el ictus isquemico, a partir de la anatomia
de la lesion y mediante in-context learning. El proyecto tiene dos etapas que se
entrenan en secuencia y luego se componen:

- Etapa 1: dos autoencoders variacionales convolucionales 3D comprimen una
  mascara de lesion y un mapa de disconectoma en un codigo compacto.
- Etapa 2: un transformer entrenado desde cero con la metodologia de
  prior-fitted network sobre una cohorte sintetica con resultados
  contrafactuales conocidos (el Neuro-Prior), que devuelve para cada paciente la
  distribucion del resultado potencial esperado condicional bajo tratamiento y
  bajo control. La diferencia es el efecto del tratamiento individualizado.

Un unico codigo corre en dos modos a partir del mismo fuente. El modo prototipo
funciona en CPU con datos reducidos y mascaras sinteticas, sin necesidad de los
datos reales ni del cluster. El modo completo escala a los nodos V100. Solo
cambian los valores de configuracion.

## Estructura

    configs/                  perfiles de configuracion (Hydra) para prototipo y completo
    src/neurocausalpfn/
      data/                   carga de NIfTI, transformaciones, covariables clinicas
      vae/                    VAE 3D, perdidas (BCE + Dice + KL), DAFT, exportacion
      prior/                  generador InterSynth, confusion, verificador R1/R2, cohorte
      pfn/                    tokens, mascara de atencion, cabeza CEPO-PPD, transformer, inferencia
      train/                  entrenamiento de la Etapa 1 y de la Etapa 2
      eval/                   root-PEHE, exactitud prescriptiva, cobertura
    tests/                    pruebas unitarias y prueba de humo de extremo a extremo
    scripts/                  ejecucion en prototipo y plantilla para el cluster

## Instalacion

Modo prototipo (CPU):

    conda env create -f env/environment.prototype.yml
    conda activate neuro-causal-pfn-proto
    pip install -e .

Para los datos reales en NIfTI y las lineas base causales se anaden los extras:

    pip install -e ".[imaging,baselines,cluster]"

## Ejecucion rapida (prueba de humo)

    bash scripts/run_prototype.sh

Esto entrena la Etapa 1 y la Etapa 2 en modo prototipo en CPU en segundos, con
datos sinteticos. Tambien se puede llamar a cada etapa por separado:

    python -m neurocausalpfn.train.train_vae --mode prototype
    python -m neurocausalpfn.train.train_pfn --mode prototype

## Pruebas

    pip install pytest
    PYTHONPATH=src pytest -q

Las dos pruebas mas importantes son la de la mascara de atencion (que el peso de
una consulta sobre otra sea exactamente cero) y la del verificador de
identificabilidad (que acepte un proceso ignorable y rechace uno con un
confundidor no observado), porque esa ultima operacionaliza el requisito de
convergencia del prior-fitted network.

## Datos

Disposicion de carpetas (todo bajo `data/`, que esta en el `.gitignore`):

    data/
      lesions/          mascaras de lesion (lesions.zip de Giles)  -> entrada de la Etapa 1
      atlases/          parcelacion funcional y subdivisiones        -> solo si se usa InterSynth real
      disconnectomes/   opcional                                     -> representacion alternativa
      representation/   representation_{hash}.npz (Z + clinico)      -> puente Etapa 1 a Etapa 2

El dataset de lesiones (`LesionMaskDataset`) busca mascaras NIfTI en el
directorio indicado en `configs/data/lesion.yaml` (`root: data/lesions`). Si no
existen, sintetiza mascaras tipo lesion para que el prototipo corra. Las
mascaras de Giles ya estan en MNI a 91x109x91; el codigo las rellena a
96x112x96 y las binariza, asi que no hace falta mas preprocesado para el VAE.

La edad y el sexo no vienen en una tabla sino en el nombre de archivo, con el
patron `lesion{id}_{age}_{sex}.nii.gz` y el literal `NA` cuando faltan. El
parser de `data/clinical.py` los extrae y construye un vector de covariables con
indicadores de dato faltante; `LesionMaskDataset.clinical_matrix()` devuelve esa
matriz alineada con el orden de las mascaras.

## El prior de la Etapa 2: sintetico o InterSynth

El transformer se entrena sobre un prior de procesos, elegido por configuracion
en `cfg["prior"]["kind"]`:

- `synthetic`: el generador ligero (`prior/intersynth.py`), que muestrea
  covariables gaussianas desde cero. Es el de por defecto y el que usan el
  prototipo y la prueba de humo.
- `intersynth`: el mecanismo anatomico real (`prior/intersynth_atlas.py` mas
  `prior/atlas.py`), que cruza cada lesion con la parcelacion funcional para
  fabricar la verdad de terreno: deficit por solapamiento de al menos el 5% con
  una subred, susceptibilidad al tratamiento segun la subred (transcriptomica o
  receptomica) dominante, desenlace por combinacion de efecto del tratamiento y
  recuperacion espontanea, y asignacion con confusion observada (distancia del
  centroide) u opcionalmente no observada. El covariable que ve el transformer es
  el latente del encoder si se pasa `z_pool`, o las covariables observadas en su
  defecto. Para activarlo: `--prior intersynth`, con `atlas_dir` apuntando a
  `data/atlases`. El cargador lee la estructura real de Giles:
  `functional_parcellation_2mm.nii.gz` (redes etiquetadas 1..K) y
  `2mm_parcellations/{modality}/` con un archivo por red cuyas dos subredes
  son las etiquetas 1 y 2. La modalidad es `receptor` (receptoma de Hansen)
  o `genetics` (transcriptoma de Allen), elegible por configuracion.

## Notas de implementacion

- El encoder del VAE se congela tras la Etapa 1; su salida se exporta una sola
  vez y se versiona por un hash de los pesos, de modo que cada resultado de la
  Etapa 2 es trazable hasta una representacion exacta.
- El objetivo del transformer es la perdida de histograma sobre el resultado
  potencial esperado condicional verdadero, con la longitud de contexto en
  curriculo de menor a mayor.
- El esqueleto usa proyecciones lineales para las filas y un softmax estandar en
  la atencion. Para los contextos grandes del modo completo se sustituiran por la
  codificacion tabular de columna y luego fila y por una atencion mas estable.

## Puntos abiertos

Quedan por confirmar la identidad del ensayo de validacion, la escala del
objetivo de precision, el tamano del transformer (a justificar con la ablacion
de backbone), la licencia del VAE de referencia y la procedencia del
disconectoma. El detalle esta en el documento del plan de implementacion.
