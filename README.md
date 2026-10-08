# CC3045 – Laboratorio 9: VAE y difusión sobre Fashion-MNIST

Enunciado completo: [`CC3045_Laboratorio9.md`](CC3045_Laboratorio9.md).

| Parte | Estado | Dónde |
|---|---|---|
| Task 1 – VAE | ✅ hecho | `lab9.ipynb` (secciones Task 1.x) |
| Task 2 – Proceso forward | ✅ hecho | `lab9.ipynb` (secciones Task 2.x) |
| Task 3 – Difusión condicional + CFG | ⏳ pendiente | `lab9.ipynb`, ver reparto abajo |
| Task 4 – Comparación y pass@k | ⏳ pendiente | `lab9.ipynb`, ver reparto abajo |

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

Todo va en **`lab9.ipynb`**. Al ejecutar las secciones de las Tasks 1 y 2 quedan disponibles:

* `SEED = 42`, `set_seed()`, `device`, `CLASSES`.
* Datos: `X_train`, `X_val` en [0,1] y `X_train_d`, `X_val_d` en [−1,1] (55 000 / 5 000), con etiquetas `Y_train`, `Y_val`.
* Calendario lineal: `betas`, `alphas`, `alpha_bar`, `snr` (float64, CPU) y `betas_d`, `alpha_bar_d` (float32, en `device`). El índice `t-1` corresponde al instante `t`.
* `q_sample(x0, t, eps)`: forma cerrada del forward, con `t` en {1..T}.
* `vaes[beta]`: los VAE entrenados (también en `checkpoints/vae_beta*.pt`); la Task 4.1 usa `vaes[1.0]`.
* `sharpness(x)`: nitidez de la Task 1.4(d), con imágenes en [0,1].
* `results/results_t1_t2.json` con todos los números de las Tasks 1 y 2. Útil para la 3.5(d): con el calendario lineal hay 647 pasos con SNR ∈ [0.01, 100] (t = 28–674), 27 con SNR > 100 y 326 con SNR < 0.01, y el primer t con SNR < 1 es 260.

## Reglas del enunciado (aplican a todos)

* Solo tensores de PyTorch: `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.GroupNorm`, `nn.Embedding`, activaciones, `Adam`, `backward()`.
  **Prohibido** `diffusers`, `torch.distributions.kl_divergence` y calendarios/muestreadores prefabricados. El muestreo se escribe a mano.
* Semilla **42** (`set_seed()` al inicio y antes de entrenar/generar).
* Código **comentado** con la relación a las fórmulas de las diapositivas.
* Cada respuesta debe citar **nuestros** números y figuras.

## Convenciones del equipo

* **Un solo notebook: `lab9.ipynb`.** Cada quien agrega sus celdas al final, debajo de lo anterior, y no modifica las celdas de los demás.
* **Se trabaja por turnos** (persona 1 → persona 2 → persona 3), porque dos personas editando el mismo `.ipynb` a la vez generan conflictos en git. Antes de empezar: `git pull`. Al terminar: commit y push, y avisar al siguiente.
* **Un commit por subtask**, mensaje de una línea con el formato del historial: `feat(task3.1): ...`, `feat(task3.5b): ...`, `feat(task4.2): ...`. Antes, configurar la identidad propia: `git config user.name "..."` y `git config user.email "..."`.
* **No reentrenar en cada corrida:** lo costoso se guarda en disco y la celda lo carga si ya existe (`if os.path.exists(...)`). Así el siguiente puede ejecutar el notebook completo sin esperar el entrenamiento de nuevo:
  `checkpoints/unet_cfg.pt`, `checkpoints/classifier.pt` y `results/samples_w1.pt`, `samples_w3.pt`, `samples_w7.pt` (imágenes generadas en [−1,1] con sus etiquetas).
* Figuras en `figures/` con prefijo de la task (`t3_2_...png`, `t4_1_...png`) y números en `results/`.
* Respuestas en el documento del equipo con el mismo formato de las Tasks 1 y 2: párrafos normales debajo de cada inciso, con figuras y tablas.

## Reparto de las Tasks 3 y 4

### Persona 1 – Tasks 3.1, 3.2, 3.3 y 3.5(d) (modelo, entrenamiento y muestreo)
* **3.1** U-Net pequeña: nivel 28×28 con 32 canales, nivel 14×14 con 64, cuello de botella a 14×14, subida con skips concatenados y conv final a 1 canal. Embedding sinusoidal de t (fórmula del positional encoding de la Semana 6) + MLP; clase con `nn.Embedding(11, d)` (10 = condición nula); la suma se inyecta con broadcasting en cada bloque. Reportar el número de parámetros.
* **3.2** Entrenar 30 épocas (batch 128, lr 2e-4, `p_uncond = 0.1`, t uniforme, `q_sample`, MSE entre ε y ε̂) y guardar `checkpoints/unet_cfg.pt`. Gráfica de pérdida por época y pérdida de validación en 10 intervalos de t de 100 pasos.
* **3.3** Muestreo ancestral escrito a mano (σ²_t = β_t, z = 0 en el último paso); cuadrícula 10×10 con w = 1, recortando a [−1,1]. Escribirlo como una función reutilizable que reciba la predicción de ruido, porque la persona 2 le agrega la guía.
* **3.5(d)** Intervalo de t con pérdida mayor y menor en su gráfica de la 3.2, relacionado con el SNR y con la predicción de la Task 2.5(b): máxima en t ∈ [1,100] y casi nula para t > 700. Si necesita una celda de apoyo, va al final de sus celdas, después de la 3.3.
* Es el camino crítico: conviene alguien con GPU (o Colab con GPU) y empezar cuanto antes.

### Persona 2 – Tasks 3.4 y 3.5(a), (b), (c) (guía sin clasificador y su análisis)
* **3.4** Guía sin clasificador sobre el muestreo de la persona 1 (w = 1 equivale al condicional puro), 50 imágenes por clase para w ∈ {1, 3, 7} (guardarlas en `results/samples_w*.pt`), clasificador con ≥ 88% en validación (`checkpoints/classifier.pt`), fidelidad, diversidad (y la de 50 reales por clase), tabla con tiempo por 100 imágenes y figura de 10 muestras de una clase por cada w.
* **3.5(a)** Cambio de fidelidad y diversidad al aumentar w; clase cuya diversidad cae más entre w = 1 y w = 7 y una hipótesis respaldada por los datos.
* **3.5(b)** Matriz de confusión del clasificador sobre las imágenes generadas con w = 1, par más confundido y si es defecto del generador, del clasificador o de las clases (con imágenes).
* **3.5(c)** Demostración de que la guía corresponde a p̃(x|c) ∝ p(x) p(c|x)^w (score s ≈ −ε_θ/√(1 − ᾱ_t) y Bayes sobre gradientes de logaritmos) y qué pasa con la diversidad al crecer w.
* Mientras la persona 1 entrena, puede ir entrenando el clasificador (solo usa imágenes reales) y escribiendo la demostración (c), en un borrador que agrega al notebook en su turno.

### Persona 3 – Task 4 (comparación y pass@k)
* **4.1** Nitidez (`sharpness`, en [0,1]: difusión con `(x+1)/2`) de 500 imágenes de difusión con w = 1 (las de la 3.4), 500 del VAE con β = 1 y 500 reales; tiempo de generar 100 imágenes con el VAE (1 evaluación de red) y con la difusión sin guía (1 000), y su razón; ubicación de cada modelo en el mapa nitidez / velocidad / estabilidad y cómo acortar la difusión.
* **4.2** pass@k: estimador insesgado para k = 1, 5, 10 (pasos de P2 con k = 5), fórmula de P2 y menor k con pass@k > 0.9, comparación y demostración frente al estimador ingenuo, y el análisis del agente con 10 reintentos.
* La 4.2 no depende de nadie: se puede resolver desde ya (en un borrador) y agregar al notebook en su turno. La 4.1 necesita las muestras de la persona 2.

## Herramientas

`tools/` tiene los scripts con los que se insertaron las respuestas de las Tasks 1 y 2 en el documento Word (`tools/fill_docx.py`, que usa `tools/docx_helpers.py`, la plantilla `tools/Lab9_DL_plantilla.docx` y `results/results_t1_t2.json`). No son necesarios para resolver las tasks.
