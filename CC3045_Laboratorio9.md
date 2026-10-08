# CC3045 – Deep Learning
# Laboratorio 9

## Instrucciones

En este laboratorio usted implementa desde cero dos de las familias de modelos generativos vistas en clase, el autoencoder variacional y el modelo de difusión, sobre el conjunto Fashion-MNIST. Además de construir los modelos, verifica numéricamente resultados que en clase se derivaron en la pizarra: la forma cerrada del proceso forward, la divergencia KL entre gaussianas, la guía sin clasificador y el estimador de pass@k.

Cada respuesta de análisis debe apoyarse en los números y las figuras que usted produjo. Una respuesta que podría escribirse sin ejecutar el código no recibirá puntos.

> **NOTA:** Todo el código debe estar escrito con tensores de PyTorch. Puede usar `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.GroupNorm`, `nn.Embedding`, funciones de activación, `torch.optim.Adam` y `loss.backward()`. No se permite la librería `diffusers`, `torch.distributions.kl_divergence` ni calendarios o muestreadores prefabricados. Las fórmulas del VAE, del proceso forward y del muestreo deben estar escritas por usted. Fije una semilla aleatoria y repórtela.

---

## Task 1 (Entrega Parcial)

### Task 1.1

Descargue Fashion-MNIST, que contiene 60 000 imágenes de entrenamiento y 10 000 de prueba, de $28 \times 28$ píxeles en escala de grises y 10 clases, con la siguiente instrucción:

```python
from torchvision import datasets
datasets.FashionMNIST("data", train=True, download=True)
```

Escale los píxeles al intervalo $[0,1]$, separe los últimos 5 000 ejemplos del conjunto de entrenamiento como validación y utilice los siguientes valores de configuración:

```python
batch_size  = 128
latent_dim  = 2
epochs      = 20
lr          = 1e-3
beta_values = [0.1, 1.0, 10.0]
```

### Task 1.2

Implemente un autoencoder variacional con encoder y decoder a su elección (completamente conectados o convolucionales), con la misma arquitectura para todos los valores de $\beta$. El encoder debe producir dos vectores de dimensión `latent_dim`: la media $\mu$ y el logaritmo de la varianza $\log\sigma^2$. Escriba usted mismo, sin funciones de distribuciones de PyTorch:

a. La reparametrización $z = \mu + \sigma \odot \epsilon$, con $\epsilon \sim \mathcal{N}(0, I)$.

b. La divergencia KL en forma cerrada entre $q(z \mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ y $\mathcal{N}(0, I)$, sumada sobre las dimensiones latentes.

c. La pérdida total por imagen, con el error cuadrático sumado sobre los 784 píxeles:

$$\mathcal{L}_\beta = \lVert x - \hat{x} \rVert^2 + \beta \, D_{KL}\big(q(z \mid x) \,\Vert\, p(z)\big)$$

Entrene un modelo para cada valor de `beta_values`. Grafique, para cada $\beta$, las curvas de validación del error de reconstrucción y de la KL por época.

### Task 1.3

Con los tres modelos entrenados, genere las siguientes figuras:

a. Un diagrama de dispersión de $\mu$ para las 5 000 imágenes de validación, coloreado por clase, para cada $\beta$ (tres paneles).

b. Para $\beta = 1$, las imágenes decodificadas sobre una cuadrícula regular de $15 \times 15$ puntos de $z$ en $[-3, 3]^2$.

c. Para cada $\beta$, 64 imágenes generadas con $z \sim \mathcal{N}(0, I)$.

d. Para $\beta = 1$, la interpolación entre dos imágenes de validación de clases distintas, con 8 puntos intermedios: una vez en el espacio de píxeles y otra vez en el espacio latente (sobre las medias $\mu$ de ambas imágenes).

### Task 1.4

Considere y conteste con base en sus propios resultados:

a. Construya una tabla con el error de reconstrucción y la KL de validación finales para cada $\beta$. Convierta la KL a bits (divida entre $\ln 2$) y compárela con los $\log_2 10 \approx 3.32$ bits necesarios para identificar la clase de una imagen. ¿Qué concluye sobre la información que el código latente de $\beta = 10$ puede transmitir sobre la clase? Contraste su conclusión con el diagrama de dispersión de $\beta = 10$.

b. Verifique su KL en forma cerrada. Tome una imagen de validación, obtenga su $\mu$ y $\sigma$ con el modelo de $\beta = 1$, estime la KL por Monte Carlo con $10^5$ muestras de $q(z \mid x)$ como el promedio de $\log q(z) - \log p(z)$, y reporte ambos valores y el error relativo. ¿Es el error compatible con la variabilidad esperada de una estimación Monte Carlo de ese tamaño?

c. Con su cuadrícula y sus interpolaciones, identifique una región del espacio latente donde las imágenes decodificadas no son reconocibles y explíquela en términos del término KL y del valor de $\beta$. Describa qué se observa en el punto medio de la interpolación en píxeles y en el de la interpolación latente, y por qué difieren.

d. Mida la nitidez como la magnitud media de la diferencia entre píxeles horizontalmente adyacentes, sobre 1 000 imágenes reales y sobre 1 000 imágenes generadas por cada $\beta$, y reporte los cuatro valores. Si el decoder es gaussiano con varianza $\sigma_x^2$, la pérdida negativa del ELBO es proporcional a $\lVert x - \hat{x} \rVert^2 + 2\sigma_x^2 \, D_{KL}$. Muestre que su $\mathcal{L}_\beta$ corresponde a $\sigma_x^2 = \beta/2$ y use esa relación, junto con sus valores de nitidez, para explicar por qué las muestras del VAE salen borrosas.

---

## Task 2 (Entrega Parcial)

### Task 2.1

Construya el proceso forward de difusión sobre Fashion-MNIST escalada al intervalo $[-1, 1]$, con los siguientes valores:

```python
T          = 1000
beta_start = 1e-4
beta_end   = 0.02
```

Calcule el calendario lineal de $\beta_t$ para $t = 1, \dots, T$, junto con $\alpha_t = 1 - \beta_t$ y $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Implemente una función que, dado un lote $x_0$, un vector de instantes $t$ y un ruido $\epsilon$, devuelva $x_t$ con la fórmula cerrada vista en clase:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

### Task 2.2

Verifique empíricamente la fórmula cerrada. Realice:

a. Fije una imagen $x_0$. Cree 5 000 copias y aplique a todas, durante $t = 300$ pasos, el proceso paso a paso $x_s = \sqrt{\alpha_s}\, x_{s-1} + \sqrt{\beta_s}\, \epsilon_s$, con ruido independiente en cada paso y en cada copia.

b. Calcule, por píxel, la media y la varianza empíricas de las 5 000 copias en $t = 300$, y compárelas con los valores teóricos $\sqrt{\bar{\alpha}_{300}}\, x_0$ y $1 - \bar{\alpha}_{300}$. Reporte el error máximo absoluto de la media y el promedio de la varianza empírica frente a la teórica.

c. Indique si el error de la media es compatible con el error estándar esperado de un promedio de 5 000 muestras, y justifique con un cálculo.

### Task 2.3

Realice:

a. Visualice $x_t$ para $t \in \{1, 50, 100, 250, 500, 750, 1000\}$ para una imagen de cada una de tres clases de su elección.

b. Grafique $\bar{\alpha}_t$ y la relación señal a ruido $\mathrm{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$ en función de $t$, con el SNR en escala logarítmica.

c. Reporte el primer $t$ con $\mathrm{SNR}(t) < 1$ y el primer $t$ con $\bar{\alpha}_t < 0.01$.

### Task 2.4

Diseñe un segundo calendario propio, con cualquier forma funcional para $\beta_t$ o para $\bar{\alpha}_t$, que cumpla $\bar{\alpha}_T < 10^{-3}$ y $\beta_t \in (0, 1)$ para todo $t$. Si parte de uno de la literatura, cítelo y modifíquelo. Superponga su curva de SNR a la del calendario lineal y reporte, para ambos calendarios, el valor de $\bar{\alpha}_T$ y el primer $t$ con $\mathrm{SNR}(t) < 1$.

### Task 2.5

Considere y conteste:

a. Para el calendario lineal, use la aproximación

   $$\log \bar{\alpha}_T = \sum_{t=1}^{T} \log(1 - \beta_t) \approx -\sum_t \beta_t - \tfrac{1}{2} \sum_t \beta_t^2.$$

   Calcule a mano la primera suma (suma aritmética) y la segunda (aproximación integral de la progresión lineal), obtenga un valor aproximado de $\bar{\alpha}_T$ y compárelo con el valor numérico de su código. Muestre los pasos.

b. Cuente, para cada calendario, cuántos de los $T$ pasos tienen $\mathrm{SNR}(t) \in [0.01, 100]$. En los pasos con $\mathrm{SNR} \ll 0.01$ se cumple $x_t \approx \epsilon$, y en los de $\mathrm{SNR} \gg 100$ se cumple $x_t \approx x_0$. Explique, para cada régimen, qué tan difícil es predecir $\epsilon$ a partir de $x_t$ y cuánto aporta ese paso al aprendizaje cuando $t$ se sortea de forma uniforme. Con base en su conteo, indique cuál de sus dos calendarios es más eficiente.

c. Demuestre que si $\mathrm{Var}(x_0)$ es la varianza global de los píxeles de $x_0$, entonces la varianza global de $x_t$ es $\bar{\alpha}_t \mathrm{Var}(x_0) + (1 - \bar{\alpha}_t)$. Compruébelo numéricamente: calcule $\mathrm{Var}(x_0)$ sobre el conjunto de entrenamiento escalado a $[-1, 1]$ y compare la varianza empírica de $x_t$ con la predicción para $t \in \{1, 250, 500, 1000\}$. ¿Qué varianza tiene $x_T$ y por qué es necesario que sea ese valor para generar desde $\mathcal{N}(0, I)$?

---

## Task 3 (Entrega Final)

### Task 3.1

Implemente una U-Net pequeña que reciba una imagen ruidosa $x_t$ de $1 \times 28 \times 28$, el instante $t$ y una clase $c \in \{0, \dots, 9, 10\}$ (la clase 10 representa la condición nula), y devuelva una predicción del ruido con la misma forma que $x_t$. Debe tener dos niveles de resolución: uno a $28 \times 28$ con 32 canales y otro a $14 \times 14$ con 64 canales, un cuello de botella a $14 \times 14$, un camino de subida con conexiones de salto concatenadas y una convolución final a un canal.

El instante entra mediante un embedding sinusoidal, que usted debe implementar con la misma fórmula del positional encoding de la Semana 6, seguido de un MLP. La clase entra mediante `nn.Embedding(11, d)`. La suma de ambos vectores se inyecta, con broadcasting, en las características de cada bloque. Reporte el número total de parámetros.

### Task 3.2

Entrene el modelo condicional con el calendario de la Task 2.1 y los siguientes valores:

```python
epochs     = 30
batch_size = 128
lr         = 2e-4
p_uncond   = 0.1
```

En cada iteración, para cada imagen del lote, sortee $t$ de forma uniforme en $\{1, \dots, T\}$ y $\epsilon \sim \mathcal{N}(0, I)$, construya $x_t$ con su función de la Task 2.1, reemplace la clase por la condición nula con probabilidad `p_uncond` y minimice el error cuadrático medio entre $\epsilon$ y la predicción de la red. Grafique la pérdida de entrenamiento por época. Al terminar, evalúe sobre las 5 000 imágenes de validación la pérdida promedio en cada uno de 10 intervalos de $t$ de 100 pasos y grafíquela contra el intervalo.

### Task 3.3

Implemente el muestreo ancestral. Parta de $x_T \sim \mathcal{N}(0, I)$ y, para $t = T, \dots, 1$, aplique:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \hat{\epsilon} \right) + \sigma_t z, \qquad \sigma_t^2 = \beta_t, \qquad z \sim \mathcal{N}(0, I)$$

con $z = 0$ en el último paso y $\hat{\epsilon}$ la predicción de la red. Genere 10 imágenes por clase usando únicamente la predicción condicional ($w = 1$) y muéstrelas en una cuadrícula de 10 filas, una por clase, recortando los valores al intervalo $[-1, 1]$.

### Task 3.4

Implemente la guía sin clasificador. Realice:

a. Implemente la predicción guiada, con la convención en la que $w = 1$ equivale al modelo condicional puro, e incorpórela al muestreo de la Task 3.3:

   $$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)$$

b. Genere 50 imágenes por clase para cada $w \in \{1, 3, 7\}$.

c. Entrene un clasificador sencillo sobre las imágenes reales de entrenamiento hasta alcanzar al menos 88% de exactitud en validación, y úselo para medir la fidelidad: la fracción de imágenes generadas que el clasificador asigna a la clase solicitada. Mida la diversidad como la distancia euclidiana promedio entre pares de imágenes generadas de la misma clase, promediada sobre las clases. Calcule también la diversidad de 50 imágenes reales por clase como referencia.

d. Presente una tabla con $w$, fidelidad global, diversidad global y tiempo de generación por cada 100 imágenes, y una figura con 10 muestras de una clase de su elección para cada $w$.

### Task 3.5

Considere y conteste con base en sus propios resultados:

a. Describa cómo cambian la fidelidad y la diversidad al aumentar $w$. Identifique la clase cuya diversidad cae más entre $w = 1$ y $w = 7$ y proponga una hipótesis, respaldada por sus datos, de por qué esa clase es la más afectada.

b. Calcule la matriz de confusión del clasificador sobre las imágenes generadas con $w = 1$ e indique el par de clases que más se confunde. ¿Esa confusión es un defecto del modelo generativo, del clasificador o de las propias clases? Respalde su respuesta con las imágenes.

c. Demuestre que la predicción guiada corresponde a muestrear de una distribución $\tilde{p}(x \mid c) \propto p(x)\, p(c \mid x)^w$. Use que el score cumple $s \approx -\epsilon_\theta / \sqrt{1 - \bar{\alpha}_t}$ y la regla de Bayes aplicada a los gradientes de los logaritmos. Explique con ella qué ocurre con la diversidad cuando $w$ crece.

d. En su gráfica de la pérdida por intervalo de $t$ de la Task 3.2, indique en qué intervalo la pérdida es mayor y en cuál es menor, y relacione su respuesta con el SNR y con su análisis de la Task 2.5(b). ¿Coincide el comportamiento con lo que usted predijo?

---

## Task 4 (Entrega Final)

### Task 4.1

Compare las dos familias con sus propios resultados. Realice:

a. Mida la nitidez definida en la Task 1.4(d) sobre 500 imágenes generadas por difusión con $w = 1$ (las de la Task 3.4), sobre 500 generadas por su VAE con $\beta = 1$ y sobre 500 imágenes reales. Reporte los tres valores.

b. Mida el tiempo de generación de 100 imágenes con su VAE y con su difusión sin guía, y reporte cuántas evaluaciones de red requiere cada una. Calcule la razón de tiempos.

c. Con base en a y b, explique en qué punto del mapa de compensaciones visto en clase (nitidez, velocidad y estabilidad de entrenamiento) se ubica cada una de sus dos implementaciones, y qué cambiaría usted para acortar el tiempo de la difusión.

### Task 4.2

Un modelo de código generó $n = 20$ soluciones para cada uno de seis problemas de programación, y $c$ de ellas pasaron todas las pruebas:

| Problema | $n$ | $c$ |
|----------|-----|-----|
| P1       | 20  | 0   |
| P2       | 20  | 2   |
| P3       | 20  | 5   |
| P4       | 20  | 9   |
| P5       | 20  | 14  |
| P6       | 20  | 20  |

Considere y conteste:

a. Calcule pass@1, pass@5 y pass@10 de cada problema con el estimador insesgado

   $$1 - \binom{n-c}{k} \Big/ \binom{n}{k},$$

   y el promedio de cada métrica sobre los seis problemas. Muestre los pasos de P2 con $k = 5$.

b. Para P2, demuestre que $\text{pass@}k = 1 - \dfrac{(20-k)(19-k)}{380}$ y encuentre el menor $k$ para el cual $\text{pass@}k > 0.9$.

c. Compare el estimador insesgado con la versión ingenua $1 - (1 - c/n)^k$ para $k = 5$ en los seis problemas. Demuestre que la versión ingenua nunca excede al estimador insesgado, usando que

   $$\frac{n - c - i}{n - i} \le \frac{n - c}{n} \quad \text{para } i \ge 0$$

   (cuando $n - c < k$, el estimador insesgado vale 1).

d. Según su tabla, ¿cuáles problemas resolvería un agente que reintenta hasta 10 veces con un verificador perfecto, y cuáles no? ¿Qué le dice la diferencia entre el promedio de pass@1 y el de pass@10 sobre este modelo?

---

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task.
2. En la entrega parcial se espera que entreguen lo **señalado**. En la entrega final deben entregar **TODOS LOS TASK**.
3. Archivo `.ipynb`, o link a repositorio de GitHub (**No se acepta entregas en otros medios**).
   - El código debe estar comentado explicando la relación con las fórmulas de las diapositivas.

## Evaluación

1. [1.00 pt] Task 1
2. [1.00 pt] Task 2
3. [1.50 pt] Task 3
4. [0.50 pt] Task 4

**Total: 4.0 pts**
