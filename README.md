# Instalación

El proyecto puede ejecutarse en **Windows** y **Linux**. La data, modelos entrenados y resultados pesados no se versionan directamente en el repositorio; deben copiarse a la carpeta `data/` en la raíz del proyecto, respetando los nombres de subcarpetas existentes.

[DATA](https://drive.google.com/drive/folders/1CvdPr07lIk7pZjJjbqGPD6GJqKHV50hb?usp=sharing)

---

## 1) Crear entorno virtual

### Windows

```cmd
py -3.13 -m venv .venv
```

### Linux

```bash
python3.13 -m venv .venv
```

---

## 2) Activar entorno virtual

### Windows

```cmd
.venv\Scripts\activate
```

### Linux

```bash
source .venv/bin/activate
```

---

## 3) Instalar dependencias

### Windows / Linux

```bash
python -m pip install -r requirements.txt
```

---

# PyTorch con GPU (CUDA)

Si vas a usar GPU, instala PyTorch con el wheel CUDA correspondiente a tu GPU, drivers y sistema. Ejemplo con `cu128`:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Si ya instalaste `torch` desde `requirements.txt`, este comando puede reemplazarlo. Esto es normal.

### Verificar que PyTorch detecta CUDA/GPU

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

> **NOTA:** Se usó e instaló PyTorch para una NVIDIA GeForce GTX 1650 con Driver Version **581.57** y CUDA Version **13.0**, en un computador con NVIDIA e Intel Graphics. La instalación exacta depende de la tarjeta gráfica, drivers y stack CUDA disponible.

---

# Compatibilidad con modelos antiguos

Los modelos antiguos siguen funcionando en modo compatible usando:

```json
{
  "node_feature_set": "basic",
  "edge_feature_mode": "none",
  "sinkhorn_nll_weight": 0.0,
  "decoder_sources": ["logits"]
}
```

Este modo conserva el comportamiento anterior:

```text
features básicas de nodo
sin atributos de arista
loss principal con cross-entropy
decoding desde logits
```

Si un checkpoint antiguo no contiene los campos nuevos, el código usa valores por defecto compatibles.

Para entrenar modelos nuevos con las mejoras actuales, se debe actualizar el JSON de entrenamiento con:

```json
{
  "node_feature_set": "ttp_v1",
  "edge_feature_mode": "distance_v1",
  "compute_dist_matrix": true,
  "sinkhorn_nll_weight": 0.05
}
```

Regla importante:

```text
Si edge_feature_mode = "distance_v1",
entonces compute_dist_matrix debe ser true.
```

**Nota** No se pudo probar la compatibilidad, así que se dejo abierta la rama "Update_code", donde aun se pueden correr los modelos anteriores.

---

# Ejecutar el proyecto

El punto de entrada principal es:

### Windows

```cmd
py main.py
```

### Linux

```bash
python main.py
```

También se puede indicar un comando:

### Windows

```cmd
py main.py <command>
```

### Linux

```bash
python main.py <command>
```

El flujo real del proyecto es:

```text
main.py
→ scripts/
→ src/ttp_packages/application/
→ paquetes internos
→ configs/
```

---

# Comandos admitidos por `main.py`

| Command           | Descripción                                                                                                          |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `generate-data`   | Genera o extiende datasets tensoriales TTP usando `configs/datasets/generate_dataset.json`.                          |
| `merge-data`      | Fusiona varios datasets en uno usando `configs/datasets/merge_dataset.json`.                                         |
| `benchmark`       | Compara CS2SA-R contra TSP+KRP usando `configs/evaluation/benchmark_cs2sar_vs_tsp.json`.                             |
| `evaluate-solver` | Evalúa CS2SA-R con distintos presupuestos de tiempo usando `configs/evaluation/evaluate_solver_cs2sar.json`.         |
| `evaluate-model`  | Evalúa un checkpoint neuronal contra solvers/baselines usando `configs/evaluation/evaluate_model_with_solvers.json`. |
| `fit-model`       | Entrena un modelo tour-only usando `configs/training/fit_tour_model.json`.                                           |
| `optuna`          | Ejecuta búsqueda de hiperparámetros con Optuna usando `configs/optuna/optuna_tour.json`.                             |

> Los archivos dentro de `scripts/` contienen la lógica CLI de cada workflow, pero la forma recomendada de ejecución es mediante `main.py`.


# Estructura del proyecto

```text
ttp_transformer/
├── configs/                                      # Config con todos los párametros de entrada para cada situación (Inputs)
│   ├── datasets/                                 # Parámetros para generar y fusionar datasets
│   │   ├── generate_dataset.json
│   │   ├── instancias.json
│   │   └── merge_dataset.json
│   ├── evaluation/                               # Evaluar modelo y solvers comparando resultados de función objetivo.
│   │   ├── benchmark_cs2sar_vs_tsp.json
│   │   ├── evaluate_model_with_solvers.json
│   │   └── evaluate_solver_cs2sar.json
│   ├── optuna/                                   # Opciones y limites de parametros para que optuna pruebe.
│   │   └── optuna_tour.json
│   └── training/                                 # Parámetros para entrenar modelo.
│       └── fit_tour_model.json
│
├── data/                                         # Datos del proyecto
│   ├── instances/                                # Instancias TTP (archivos de entrada)
│   ├── optuna/                                   # Persistencia de Optuna: estudios, trials, estados y resultados
│   ├── results_compare/                          # Plots de comparación (solver vs greedy vs modelo)
│   ├── train_data/                               # Dataset de entrenamiento (muchos samples)
│   ├── trained_history_runs/                     # Historial por run (loss/accuracy)
│   ├── trained_models/                           # Modelos guardados (entrenados)
│   ├── trained_models_params/                    # Parámetros por modelo (JSON de config/entrenamiento)
│   └── training_plots/                           # Gráficos de entrenamiento (loss/accuracy)
│
├── scripts/                                      # Scripts CLI para ejecutar workflows del proyecto
│   ├── benchmark_cs2sar_vs_tsp.py                # Compara CS2SA-R contra TSP+KRP
│   ├── evaluate_model_with_solvers.py            # Evalúa un checkpoint contra solvers/baselines
│   ├── evaluate_solver_cs2sar.py                 # Evalúa CS2SA-R con distintos budgets
│   ├── fit_tour_model.py                         # Entrena el modelo tour-only
│   ├── gen_dataset.py                            # Genera o extiende datasets tensoriales TTP
│   ├── merge_data.py                             # Genera un nuevo dataset combinando samples de otros dataset
│   ├── optuna_tour.py                            # Busca hiperparámetros con Optuna
│   └── utils.py                                  # Utilidades compartidas de scripts
│
├── src/                                          # Código fuente principal
│   └── ttp_packages/                             # Paquetes principales del proyecto TTP
│       ├── application/                          # Casos de uso y workflows de alto nivel
│       │   ├── __init__.py
│       │   ├── benchmark_solvers.py
│       │   ├── build_model.py
│       │   ├── config.py
│       │   ├── evaluate_model.py
│       │   ├── generate_dataset.py
│       │   ├── hpo_optuna.py
│       │   ├── load_instance.py
│       │   ├── merge_dataset.py
│       │   ├── solve_instance.py
│       │   └── train_model.py
│       │
│       ├── domain/                               # Entidades, solución y objetivo del problema TTP
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── entities.py
│       │   ├── instance.py
│       │   ├── objective.py
│       │   ├── solution.py
│       │   └── tour_ops.py
│       │
│       ├── evaluation/                           # Benchmarks, métricas y comparación de soluciones
│       │   ├── __init__.py
│       │   ├── baselines.py
│       │   ├── benchmarks.py
│       │   ├── compare.py
│       │   ├── config.py
│       │   ├── fixed_tour.py
│       │   └── tour_metrics.py
│       │
│       ├── generation/                           # Generación sintética de instancias TTP
│       │   ├── math/                             # Cálculos auxiliares para generación de instancias
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   ├── item_sampling.py
│       │   │   ├── knapsack_proxy.py
│       │   │   ├── rent_estimation.py
│       │   │   └── tsp_proxy.py
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── instance_generator.py
│       │
│       ├── hpo/                                  # Soporte interno para HPO con Optuna
│       │   ├── __init__.py
│       │   ├── callbacks.py
│       │   ├── config.py
│       │   ├── results.py
│       │   ├── sampling.py
│       │   └── study.py
│       │
│       ├── infrastructure/                       # Logging, runtime y soporte operativo del proyecto
│       │   ├── storage/                          # I/O, rutas y persistencia de artefactos
│       │   │   ├── __init__.py
│       │   │   ├── compare_io.py
│       │   │   ├── dataset_io.py
│       │   │   ├── file_names.py
│       │   │   ├── instance_io.py
│       │   │   ├── json_io.py
│       │   │   ├── keys.py
│       │   │   ├── model_in.py
│       │   │   ├── model_out.py
│       │   │   ├── paths.py
│       │   │   ├── paths_config.py
│       │   │   ├── plot_io.py
│       │   │   ├── runs_io.py
│       │   │   └── torch_io.py
│       │   ├── __init__.py
│       │   ├── log_format.py
│       │   ├── logging.py
│       │   └── runtime.py
│       │
│       ├── ml_data/                              # Representación y carga de datos para ML
│       │   ├── representation/                   # Conversión de instancias y soluciones a samples
│       │   │   ├── __init__.py
│       │   │   ├── instance_solution.py
│       │   │   ├── payload.py
│       │   │   ├── payload_merge.py
│       │   │   └── tour_targets.py
│       │   ├── torch/                            # Datasets, collates y samplers PyTorch
│       │   │   ├── transforms/                   # Transformaciones y augmentación de tensores
│       │   │   │   ├── __init__.py
│       │   │   │   ├── augment.py
│       │   │   │   └── coords_norm.py
│       │   │   ├── __init__.py
│       │   │   ├── collate.py
│       │   │   ├── dataset.py
│       │   │   └── samplers.py
│       │   ├── __init__.py
│       │   └── config.py
│       │
│       ├── modeling/                             # Arquitectura neuronal para predicción de tours
│       │   ├── branches/                         # Ramas especializadas de salida del modelo
│       │   │   ├── __init__.py
│       │   │   └── tour_branch.py
│       │   ├── encoders/                         # Encoders de nodos y posición
│       │   │   ├── __init__.py
│       │   │   ├── node_encoder.py
│       │   │   └── positional_encoding.py
│       │   ├── heads/                            # Cabezas de predicción del modelo
│       │   │   ├── __init__.py
│       │   │   └── edge_heatmap_head.py
│       │   ├── layers/                           # Capas diferenciables reutilizables
│       │   │   ├── __init__.py
│       │   │   └── sinkhorn.py
│       │   ├── utils/                            # Features y máscaras auxiliares del modelo
│       │   │   ├── __init__.py
│       │   │   ├── features.py
│       │   │   └── masks.py
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── ttp_model.py
│       │
│       ├── optimization/                         # Solvers clásicos e inferencia neuronal
│       │   ├── classical/                        # Heurísticas y metaheurísticas clásicas
│       │   │   ├── packing/                      # Optimización de packing con tour fijo
│       │   │   │   ├── __init__.py
│       │   │   │   ├── api.py
│       │   │   │   ├── config.py
│       │   │   │   ├── initial_packing.py
│       │   │   │   └── local_search.py
│       │   │   ├── tsp/                          # Resolución auxiliar del TSP
│       │   │   │   ├── __init__.py
│       │   │   │   ├── config.py
│       │   │   │   ├── heuristics.py
│       │   │   │   └── tsp_api.py
│       │   │   ├── ttp/                          # Solvers específicos del TTP
│       │   │   │   ├── cs2sa_r/                  # Implementación del solver CS2SA-R
│       │   │   │   │   ├── __init__.py
│       │   │   │   │   ├── api.py
│       │   │   │   │   ├── config.py
│       │   │   │   │   ├── delta_eval.py
│       │   │   │   │   ├── initializer.py
│       │   │   │   │   ├── krp_optimizer.py
│       │   │   │   │   ├── route_cache.py
│       │   │   │   │   └── tskp_optimizer.py
│       │   │   │   ├── __init__.py
│       │   │   │   ├── greedy.py
│       │   │   │   └── random_solver.py
│       │   │   └── __init__.py
│       │   ├── neural/                           # Inferencia y decodificación neuronal de tours
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   ├── decoding.py
│       │   │   └── inference.py
│       │   └── __init__.py
│       │
│       ├── training/                             # Entrenamiento, validación y callbacks del modelo
│       │   ├── __init__.py
│       │   ├── callbacks.py
│       │   ├── config.py
│       │   ├── engine.py
│       │   ├── evaluation.py
│       │   ├── losses.py
│       │   └── utils.py
│       │
│       └── __init__.py
├── .gitignore                                    # Reglas de exclusión de Git
├── main.py                                       # Punto de entrada principal del programa
├── README.md                                     # Documentación general del proyecto
└── requirements.txt                              # Lista de dependencias de Python
```