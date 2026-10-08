# CC3045 – Laboratorio 9: VAE y difusión sobre Fashion-MNIST

Enunciado completo: [`CC3045_Laboratorio9.md`](CC3045_Laboratorio9.md).

| Parte | Estado | Dónde |
|---|---|---|
| Task 1 – VAE | ✅ hecho | `lab9.ipynb` (secciones Task 1.x) |
| Task 2 – Proceso forward | ✅ hecho | `lab9.ipynb` (secciones Task 2.x) |
| Task 3 – Difusión condicional + CFG | ⏳ pendiente | ver reparto abajo |
| Task 4 – Comparación y pass@k | ⏳ pendiente | ver reparto abajo |

## Preparar el entorno

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/Mac: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # con GPU NVIDIA
# (sin GPU: pip install torch torchvision)
pip install -r requirements.txt
```

El dataset se descarga solo en `data/` (está en `.gitignore`). Sin GPU, entrenar la U-Net y muestrear 1 000 pasos es muy lento: usar Google Colab con GPU.

## Qué ya existe y se puede reutilizar

* **`common.py`** – todo lo de las Tasks 1 y 2 listo para importar (`from common import *`):
  `SEED = 42`, `set_seed()`, `CLASSES`, `load_fashion(device)` (train 55 000 / val 5 000 en [0,1]),
  `to_diffusion_range()` ([0,1] → [−1,1]), `linear_schedule()` → `(betas, alphas, alpha_bar)` (índice `t-1` = instante `t`),
  `q_sample(x0, t, eps, alpha_bar)`, `VAE` / `load_vae(beta, device)` y `sharpness(x)`.
* **`checkpoints/vae_beta{0.1,1.0,10.0}.pt`** – los VAE entrenados (la Task 4.1 usa β = 1).
* **`results/results_t1_t2.json`** – todos los números de las Tasks 1 y 2. Útil para la 3.5(d): con el calendario lineal hay 647 pasos con SNR ∈ [0.01, 100] (t = 28–674), 27 con SNR > 100 y 326 con SNR < 0.01; el primer t con SNR < 1 es 260.
* **`figures/`** – figuras de las Tasks 1 y 2.

## Reglas del enunciado (aplican a todos)

* Solo tensores de PyTorch: `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.GroupNorm`, `nn.Embedding`, activaciones, `Adam`, `backward()`.
  **Prohibido** `diffusers`, `torch.distributions.kl_divergence` y calendarios/muestreadores prefabricados. El muestreo se escribe a mano.
* Semilla **42** (`set_seed()` al inicio y antes de entrenar/generar).
* Código **comentado** con la relación a las fórmulas de las diapositivas.
* Cada respuesta debe citar **nuestros** números y figuras.

## Convenciones del equipo

* **Un notebook por persona** (para no tener conflictos en git con el `.ipynb`), en la raíz del repo:
  `lab9_task3_modelo.ipynb` (persona A), `lab9_task3_cfg.ipynb` (persona B), `lab9_task4.ipynb` (persona C).
  Cada uno empieza con `from common import *`. No editar `lab9.ipynb`.
* **Un commit por subtask**, mensaje de una línea con el formato que ya tiene el historial:
  `feat(task3.1): ...`, `feat(task3.4c): ...`, etc. Configurar la identidad propia antes:
  `git config user.name "..."` y `git config user.email "..."`.
* Hacer `git pull` antes de empezar y antes de cada push.
* Figuras en `figures/` con prefijo de la task (`t3_2_...png`, `t4_1_...png`), números en `results/` (JSON), modelos en `checkpoints/`.
* Las imágenes de difusión se guardan en **[−1, 1]**; para la nitidez se convierten a [0,1] con `(x + 1) / 2`.
* Respuestas en el documento del equipo con el mismo formato de las Tasks 1 y 2: párrafos normales debajo de cada inciso, con figuras y tablas.

## Reparto de las Tasks 3 y 4

### Persona A – Modelo de difusión (Tasks 3.1, 3.2, 3.3 y 3.5d) · `lab9_task3_modelo.ipynb`
Es el camino crítico: B y C necesitan su checkpoint, así que conviene que A tenga GPU y empiece primero.

* **3.1** U-Net pequeña: nivel 28×28 con 32 canales, nivel 14×14 con 64, cuello de botella a 14×14, subida con skips **concatenados**, conv final a 1 canal.
  Embedding sinusoidal de t (misma fórmula del positional encoding de la Semana 6) + MLP; clase con `nn.Embedding(11, d)` (10 = condición nula); la suma se inyecta con broadcasting en cada bloque. Reportar el número de parámetros.
  La clase U-Net debe agregarse a **`common.py`** (con su función de embedding) para que B y C la importen.
* **3.2** Entrenar: 30 épocas, batch 128, lr 2e-4, `p_uncond = 0.1`, t uniforme en {1..T}, `q_sample` de `common.py`, MSE(ε, ε̂). Gráfica de pérdida por época y pérdida de validación (5 000 imágenes) en 10 intervalos de t de 100 pasos.
* **3.3** Muestreo ancestral (fórmula del enunciado, σ²_t = β_t, z = 0 en el último paso), 10 imágenes por clase con w = 1 en cuadrícula 10×10, recorte a [−1, 1].
  Escribir el muestreador como función reutilizable, p. ej. `sample(model, labels, w=1.0)`, que B extiende con la guía, y agregarlo a `common.py`.
* **3.5(d)** Intervalo de t con pérdida mayor y menor, relacionado con el SNR y con la predicción de la Task 2.5(b) (máxima en t ∈ [1,100], casi nula para t > 700).
* **Entrega a los demás:** `checkpoints/unet_cfg.pt`, U-Net y `sample()` en `common.py`, `results/t3_2_loss.json`, figuras `t3_2_*`, `t3_3_*`.

### Persona B – Guía sin clasificador y su análisis (Tasks 3.4 a, b, d y 3.5 a, b) · `lab9_task3_cfg.ipynb`
Necesita el checkpoint y el muestreador de A, y el clasificador de C. Mientras tanto puede escribir y probar el código de las métricas con imágenes reales.

* **3.4(a)** Predicción guiada ε̃ = ε(x_t, t, ∅) + w(ε(x_t, t, c) − ε(x_t, t, ∅)), con w = 1 equivalente al condicional puro, dentro del muestreo de A.
* **3.4(b)** 50 imágenes por clase para w ∈ {1, 3, 7} (1 500 imágenes, 1 000 pasos cada una: generar en lotes en GPU). Medir el tiempo por cada 100 imágenes.
  Guardar `results/samples_w1.pt`, `samples_w3.pt` y `samples_w7.pt` (tensores 500×1×28×28 en [−1,1] y sus etiquetas), porque C los usa en la 4.1.
* **3.4(c, parte de métricas)** Fidelidad = % de generadas que el clasificador de C asigna a la clase pedida; diversidad = distancia euclidiana promedio entre pares de la misma clase, promediada sobre clases; referencia con 50 imágenes reales por clase.
* **3.4(d)** Tabla (w, fidelidad, diversidad, tiempo por 100 imágenes) y figura con 10 muestras de una clase para cada w.
* **3.5(a)** Cambio de fidelidad y diversidad con w; clase cuya diversidad cae más entre w = 1 y w = 7 y una hipótesis apoyada en los datos.
* **3.5(b)** Matriz de confusión del clasificador sobre las generadas con w = 1, par más confundido y si es defecto del generador, del clasificador o de las clases (mostrar imágenes).
* **Entrega:** `results/samples_w*.pt`, `results/t3_4_metrics.json`, figuras `t3_4_*`, `t3_5_*`.

### Persona C – Clasificador, demostración de CFG y Task 4 (Tasks 3.4c clasificador, 3.5c, 4.1 y 4.2) · `lab9_task4.ipynb`
Puede empezar desde ya: el clasificador, la 3.5(c) y la 4.2 no dependen de nadie.

* **3.4(c, clasificador)** Clasificador sencillo (p. ej. CNN pequeña) entrenado con las imágenes reales **en [−1, 1]**, hasta ≥ 88% de exactitud en validación. Agregar la clase a `common.py` y guardar `checkpoints/classifier.pt` lo antes posible, porque B lo necesita.
* **3.5(c)** Demostrar que la predicción guiada corresponde a p̃(x|c) ∝ p(x) p(c|x)^w, usando s ≈ −ε_θ/√(1 − ᾱ_t) y Bayes sobre los gradientes de los logaritmos; explicar qué pasa con la diversidad cuando w crece (contrastar con la tabla de B).
* **4.1(a)** Nitidez (`sharpness` de `common.py`, imágenes en [0,1]) de 500 imágenes de difusión con w = 1 (las de B), 500 del VAE con β = 1 (`load_vae(1.0, device)`) y 500 reales.
* **4.1(b)** Tiempo de generar 100 imágenes con el VAE (1 evaluación de red) y con la difusión sin guía (1 000 evaluaciones), y la razón de tiempos.
* **4.1(c)** Ubicar ambos modelos en el mapa nitidez / velocidad / estabilidad visto en clase y proponer cómo acortar la difusión (p. ej. DDIM o menos pasos).
* **4.2** pass@k: no requiere modelos. Estimador insesgado para k = 1, 5, 10 (pasos de P2 con k = 5), demostración de la fórmula de P2 y menor k con pass@k > 0.9, comparación con el estimador ingenuo y su demostración, y el análisis del agente con 10 reintentos.
* **Entrega:** `checkpoints/classifier.pt`, clase en `common.py`, `results/t4_*.json`, figuras `t4_*`.

### Orden sugerido
1. A entrena la U-Net (3.1–3.2) mientras C entrena el clasificador y hace la 3.5(c) y la 4.2, y B prepara el código de las métricas.
2. A publica el checkpoint y `sample()` en `common.py`; C publica el clasificador.
3. B genera las 1 500 muestras y calcula las métricas; A termina la 3.3 y la 3.5(d).
4. C hace la 4.1 con las muestras de B.

## Herramientas

`tools/` tiene los scripts con los que se insertaron las respuestas de las Tasks 1 y 2 en el documento Word (`tools/fill_docx.py`, que usa `tools/docx_helpers.py`, la plantilla `tools/Lab9_DL_plantilla.docx` y `results/results_t1_t2.json`). No son necesarios para resolver las tasks.
