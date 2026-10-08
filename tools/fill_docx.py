"""Inserta las respuestas de las Tasks 1 y 2 en Lab9_DL.docx (parte de la copia original de la plantilla).
Uso: python fill_docx.py <original.docx> <destino.docx> <raíz del repo>
Texto: **negrita**, $latex$ en línea; c.eq(latex) para ecuaciones centradas."""
import json, math, sys
import docx
from docx_helpers import Cursor, find, style_run

SRC, DST, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(f"{ROOT}/results/results_t1_t2.json", encoding="utf8"))
doc = docx.Document(SRC)
C = lambda anchor: Cursor(doc, anchor, f"{ROOT}/figures")
F = lambda prefix: find(doc, prefix)
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"

# ================================================================ encabezado
repo = F("Repo:")
style_run(repo.add_run("https://github.com/Branuvg/lab9_deep"))
c = C(repo)
c.para("Semilla aleatoria **SEED = 42** (fijada en random, numpy y torch, CPU y CUDA, y re-fijada antes de entrenar cada modelo). "
       "Código: lab9.ipynb. Entorno: PyTorch 2.11 + CUDA 12.8, GPU NVIDIA RTX 5060 Laptop.", indent=False)

# ================================================================ Task 1.1
c = C(F("Escale los píxeles")._p.getnext())          # tabla con la configuración
c.para("Fashion-MNIST se descargó con torchvision.datasets.FashionMNIST y los píxeles se escalaron a $[0,1]$ dividiendo entre 255. "
       "Las primeras 55 000 imágenes de entrenamiento se usan para entrenar y las últimas 5 000 (índices 55 000–59 999) como validación; "
       "el conjunto de prueba no se usa. Se usó exactamente la configuración pedida: batch_size = 128, latent_dim = 2, epochs = 20, "
       "lr = 1e-3 (Adam) y $\\beta\\in\\{0.1,1,10\\}$.")

# ================================================================ Task 1.2
c = C(F("Entrene un modelo para cada valor"))
c.para("VAE convolucional con la misma arquitectura para los tres valores de $\\beta$ (1 677 509 parámetros). "
       "**Encoder** $q_\\phi(z\\mid x)$: Conv(1→32, stride 2, 28→14) → Conv(32→64, 14→7) → Linear(3136→256) → dos cabezas Linear(256→2) que producen $\\mu$ y $\\log\\sigma^2$. "
       "**Decoder** $p_\\theta(x\\mid z)$: Linear(2→256→3136) → ConvT(64→32, 7→14) → ConvT(32→1, 14→28) → sigmoide; su salida $\\hat{x}$ es la media del decoder, en $[0,1]$.")
c.para("Fórmulas escritas a mano (funciones reparametrize, kl_closed_form y vae_loss del notebook):")
c.eq(r"\text{(a)}\quad z=\mu+\sigma\odot\epsilon,\quad \sigma=\exp\left(\tfrac{1}{2}\log\sigma^2\right),\quad \epsilon\sim\mathcal{N}(0,I)")
c.eq(r"\text{(b)}\quad D_{KL}\left(q(z\mid x)\Vert p(z)\right)=\frac{1}{2}\sum_{j=1}^{2}\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right)")
c.eq(r"\text{(c)}\quad \mathcal{L}_\beta=\sum_{i=1}^{784}\left(x_i-\hat{x}_i\right)^2+\beta\,D_{KL}")
c.para("La pérdida (c) se calcula por imagen y se promedia sobre el lote. Cada modelo se entrenó desde la misma inicialización (semilla 42) durante 20 épocas; "
       "el entrenamiento de los tres tomó 184 s en la GPU.")
c.image("t1_2_curvas_validacion.png", 6.2, "Figura 1. Curvas de validación del error de reconstrucción y de la KL por época para cada β.")
c.para("Las curvas muestran el compromiso que controla $\\beta$. Al aumentar $\\beta$ la KL baja (≈ 8.0 nats con $\\beta=0.1$, ≈ 5 nats con $\\beta=1$ y ≈ 1.7 nats con $\\beta=10$) "
       "y el error de reconstrucción sube (≈ 20.4, 21.4 y 35 de SSE por imagen). Con $\\beta=0.1$ y $\\beta=1$ la reconstrucción baja de forma casi monótona. "
       "La de $\\beta=10$ oscila entre 34.5 y 40 porque su encoder usa $\\sigma\\approx0.4$–$0.5$, de modo que el $z$ muestreado en validación es muy ruidoso. "
       "La KL se estabiliza desde las primeras épocas: lo que mejora durante el entrenamiento es sobre todo la reconstrucción.")

# ================================================================ Task 1.3
ip = R["interp"]
c = C(F("Para , la interpolación"))
c.image("t1_3a_dispersion_mu.png", 6.5, "Figura 2 (a). μ de las 5 000 imágenes de validación coloreado por clase, para β = 0.1, 1 y 10.")
c.para("Con $\\beta=0.1$ los $\\mu$ ocupan un rango amplio ($\\mu_1\\in[-4.3,3.4]$, $\\mu_2\\in[-4.2,4.3]$) con $\\sigma$ medio ≈ 0.02: cada imagen se codifica casi como un punto. "
       "Con $\\beta=1$ el rango baja a ≈ $[-3,3.4]$ y $\\sigma\\approx0.09$. Con $\\beta=10$ se comprime a ≈ $[-2.2,3.1]$ con $\\sigma\\approx0.39$–$0.48$: "
       "cada código es una nube del tamaño de la separación entre clases.")
c.image("t1_3b_cuadricula_beta1.png", 5.0, "Figura 3 (b). β = 1: imágenes decodificadas sobre la cuadrícula 15×15 de z ∈ [−3, 3]².")
c.image("t1_3c_muestras.png", 6.5, "Figura 4 (c). 64 muestras z ~ N(0, I) por modelo (los mismos z para los tres β).")
c.image("t1_3d_interpolacion.png", 6.5, "Figura 5 (d). β = 1: interpolación Trouser → Ankle boot con 8 puntos intermedios. Arriba en píxeles, abajo en el espacio latente (sobre μ).")
c.para(f"Los extremos son las imágenes de validación {ip['idx_A']} (Trouser, $\\mu_A=({ip['muA'][0]:.2f},\\,{ip['muA'][1]:.2f})$) "
       f"y {ip['idx_B']} (Ankle boot, $\\mu_B=({ip['muB'][0]:.2f},\\,{ip['muB'][1]:.2f})$). "
       "En píxeles se usa $x_\\lambda=(1-\\lambda)x_A+\\lambda x_B$ y en el latente $\\hat{x}_\\lambda=\\mathrm{dec}\\left((1-\\lambda)\\mu_A+\\lambda\\mu_B\\right)$, con $\\lambda=0,\\tfrac19,\\dots,1$.")

# ================================================================ Task 1.4 (a)
t = {str(r["beta"]): r for r in R["t1_4a"]}
knn = R["knn_mu"]
c = C(F("Construya una tabla con el error"))
c.table(["β", "Recon. ‖x − x̂‖² (val.)", "KL (nats)", "KL (bits) = KL / ln 2", "Comparación con log₂10 ≈ 3.32 bits"],
        [[b, f3(t[b]["recon"]), f3(t[b]["kl_nats"]), f3(t[b]["kl_bits"]), "mayor" if t[b]["kl_bits"] > 3.32 else "**menor**"] for b in ["0.1", "1.0", "10.0"]])
c.para("La KL promedio es una cota superior de la información que el código transmite sobre la imagen:")
c.eq(r"\mathbb{E}_x\left[D_{KL}\left(q(z\mid x)\Vert p(z)\right)\right]=I(x;z)+D_{KL}\left(q(z)\Vert p(z)\right)\ge I(x;z)\ge I(c;z)")
c.para("La última desigualdad vale porque la clase $c$ es función de la imagen $x$. Para $\\beta=10$ la KL final es 1.858 nats = **2.68 bits < 3.32 bits**: "
       "el código latente no puede llevar la información necesaria para identificar la clase entre 10 opciones equiprobables. "
       "Aun en el mejor caso quedan al menos $H(c\\mid z)\\ge3.32-2.68=0.64$ bits de incertidumbre. "
       "Con $\\beta=0.1$ (11.9 bits) y $\\beta=1$ (7.4 bits) esa cota sí permitiría identificar la clase.")
c.para(f"El diagrama de dispersión de $\\beta=10$ (Figura 2, derecha) lo confirma. Solo quedan aisladas algunas clases de forma muy distinta (Trouser abajo a la derecha, Bag arriba, Ankle boot), "
       f"mientras que T-shirt, Shirt, Pullover, Coat y Dress se apilan sobre un mismo brazo diagonal, y Sneaker y Sandal se superponen. Además, con $\\sigma\\approx0.4$–$0.5$ cada código "
       f"es una nube que tapa a sus vecinos, así que el $z$ que recibe el decoder mezcla clases. Como medida de apoyo, un kNN ($k=15$) que predice la clase a partir de $\\mu$ "
       f"logra {100*knn['10.0']:.1f}% con $\\beta=10$, frente a {100*knn['1.0']:.1f}% con $\\beta=1$ y {100*knn['0.1']:.1f}% con $\\beta=0.1$. Esa cifra incluso es optimista, porque usa $\\mu$ sin el ruido $\\sigma$. "
       "Que $\\beta=0.1$ no supere a $\\beta=1$ pese a tener 4.5 bits más indica que el exceso de información describe detalles de la imagen (intensidad, grosor) y no la clase.")

# ================================================================ Task 1.4 (b)
b = R["t1_4b"]
c = C(F("Verifique su KL en forma cerrada"))
c.para(f"Se tomó la imagen de validación 0 (clase {b['class']}). Con el modelo $\\beta=1$: $\\mu=({b['mu'][0]:.4f},\\,{b['mu'][1]:.4f})$ y "
       f"$\\sigma=({b['sigma'][0]:.4f},\\,{b['sigma'][1]:.4f})$. Se muestrearon $N=10^5$ valores $z\\sim q(z\\mid x)$ con la reparametrización y se promedió "
       "$\\log q(z)-\\log p(z)$, con la log-densidad gaussiana diagonal escrita a mano:")
c.eq(r"\log\mathcal{N}(z;\mu,\sigma^2)=-\frac{1}{2}\sum_j\left[\log 2\pi+\log\sigma_j^2+\frac{(z_j-\mu_j)^2}{\sigma_j^2}\right],\qquad \widehat{KL}=\frac{1}{N}\sum_{i=1}^{N}\left[\log q(z_i)-\log p(z_i)\right]")
c.table(["KL forma cerrada", "KL Monte Carlo (N = 10⁵)", "|diferencia|", "Error relativo", "Error estándar MC", "|dif.| / SE"],
        [[f"{b['kl_cf']:.5f} nats", f"{b['kl_mc']:.5f} nats", f"{abs(b['kl_mc']-b['kl_cf']):.5f}", f"{100*b['rel_err']:.3f}%",
          f"{b['se']:.5f}", f"{b['n_se']:.2f}"]])
c.para(f"El error **sí es compatible** con la variabilidad Monte Carlo. La desviación estándar empírica de los términos $\\log q-\\log p$ es {b['term_std']:.4f}. "
       f"Su valor teórico para gaussianas diagonales, $\\sum_j\\left[\\tfrac12(1-\\sigma_j^2)^2+\\mu_j^2\\sigma_j^2\\right]$ bajo la raíz, da {b['term_std_theory']:.4f}. "
       f"Por lo tanto el error estándar del promedio es $SE={b['term_std']:.4f}/\\sqrt{{10^5}}={b['se']:.5f}$ nats (≈ {100*b['se']/b['kl_cf']:.3f}% relativo). "
       f"La diferencia observada equivale a {b['n_se']:.2f} SE, dentro del intervalo de ±1.96 SE (95%), así que la estimación Monte Carlo coincide con la forma cerrada dentro de su variabilidad esperada.")

# ================================================================ Task 1.4 (c)
d = R["latent_density"]; mid = R["t1_4c"]
c = C(F("Con su cuadrícula y sus interpolaciones"))
c.image("t1_4c_densidad_latente.png", 6.3, "Figura 6. β = 1: histograma de los μ de validación sobre la misma cuadrícula y trayectoria de la interpolación latente.")
c.para(f"**Regiones no reconocibles.** En la cuadrícula de $\\beta=1$ (Figura 3) hay dos zonas claras. "
       f"(1) La franja $z_2\\approx0$ con $z_1\\in[-3,-0.7]$, frontera entre Trouser (arriba) y Sneaker/Sandal (abajo), donde se decodifican trazos tenues y horizontales que no son ninguna prenda. "
       f"(2) La esquina inferior derecha ($z_1>0$, $z_2<-1.5$), donde salen rectángulos grises borrosos, una mezcla de Bag y Pullover/Coat. "
       f"El histograma de la Figura 6 muestra que ambas zonas casi no tienen códigos: {d['empty_cells']} de las 225 celdas no contienen ninguno de los 5 000 $\\mu$ de validación, "
       f"y solo {100*d['frac_r_gt2']:.1f}% de los $\\mu$ tienen $\\Vert\\mu\\Vert>2$ ({100*d['frac_r_gt25']:.1f}% con $\\Vert\\mu\\Vert>2.5$).")
c.para("**Explicación con la KL y β.** Según la fórmula (b), el término $\\beta D_{KL}$ penaliza $\\mu_j^2$ (empuja los códigos al origen) y que $\\sigma_j$ se aleje de 1. "
       "Con $\\beta=1$ los 10 grupos quedan comprimidos en un disco de radio ≈ 2, pero como $\\sigma\\approx0.09$ cada código cubre muy poco espacio. "
       "Entre grupos y en las esquinas, donde la densidad del prior $\\propto e^{-\\Vert z\\Vert^2/2}$ es muy baja en $\\Vert z\\Vert\\approx3$, quedan zonas que el decoder casi nunca vio al entrenar. "
       "Ahí extrapola o promedia a los grupos vecinos (con MSE, el decoder aprende la media condicional) y el resultado no es reconocible. "
       "$\\beta$ controla esos huecos. Con $\\beta=0.1$ la KL pesa poco, los $\\mu$ se dispersan hasta $|z|\\approx4$ con $\\sigma\\approx0.02$ y aparecen más huecos: "
       "varias de las 64 muestras de la Figura 4 son siluetas oscuras y deshechas. Con $\\beta=10$ las nubes con $\\sigma\\approx0.45$ se solapan y llenan el disco sin huecos, "
       "pero a cambio todo sale como una mezcla borrosa.")
c.image("t1_4c_puntos_medios.png", 4.3, "Figura 7. Extremos y puntos medios (λ = 0.44 y 0.56) de ambas interpolaciones.")
c.para(f"**Punto medio en píxeles:** es una doble exposición, con las dos piernas del pantalón semitransparentes encima de la silueta de la bota. No es una prenda sino la superposición de dos: "
       f"el {100*mid['pix_mid']:.0f}% de sus píxeles tiene intensidad intermedia (0.2–0.6), frente a {100*mid['real_A']:.0f}% y {100*mid['real_B']:.0f}% en las imágenes reales. "
       f"**Punto medio latente:** es un único calzado (un zapato o sandalia baja, borroso) con una sola silueta y sin fantasmas ({100*mid['lat_mid']:.0f}% de píxeles intermedios).")
c.para("**Por qué difieren.** El promedio en píxeles es un promedio euclidiano que se sale de la variedad de imágenes reales. "
       "El promedio latente, en cambio, se decodifica con un decoder entrenado para producir prendas. "
       "La trayectoria de $\\mu_A$ a $\\mu_B$ (Figura 6) cruza la frontera Trouser/Sandal ($\\lambda\\approx0.2$–$0.3$, zona de baja densidad, donde se ve una sandalia tenue) "
       "y luego el grupo de Sandal antes de llegar a Ankle boot. Por eso la interpolación latente pasa por clases semánticamente intermedias (pantalón → sandalia → bota) "
       "en lugar de mezclar dos imágenes.")

# ================================================================ Task 1.4 (d)
s = R["t1_4d"]
c = C(F("Mida la nitidez como la magnitud media"))
c.para("Se define la nitidez como la media de $|x_{i,j+1}-x_{i,j}|$ sobre todos los pares de píxeles horizontalmente adyacentes, con píxeles en $[0,1]$. "
       "Para las imágenes generadas se usa la media del decoder $\\hat{x}$ con $z\\sim\\mathcal{N}(0,I)$.")
c.table(["Imágenes (1 000 c/u)", "Nitidez", "% de la real", "σₓ² = β/2", "σₓ"],
        [["Reales (validación)", f4(s["real"]), "100%", "—", "—"]] +
        [[f"VAE β = {bb}", f4(s[f"beta={bb}"]), f"{100*s[f'beta={bb}']/s['real']:.1f}%", f"{float(bb)/2:.2f}", f"{math.sqrt(float(bb)/2):.3f}"] for bb in ["0.1", "1.0", "10.0"]])
c.para("**Demostración de $\\sigma_x^2=\\beta/2$.** Si el decoder es gaussiano, $p(x\\mid z)=\\mathcal{N}(x;\\hat{x}(z),\\sigma_x^2I)$ con $D=784$ píxeles, entonces")
c.eq(r"-\log p(x\mid z)=\frac{\Vert x-\hat{x}\Vert^2}{2\sigma_x^2}+\frac{D}{2}\log\left(2\pi\sigma_x^2\right)\quad\Rightarrow\quad -\mathrm{ELBO}=\frac{\mathbb{E}_q\Vert x-\hat{x}\Vert^2}{2\sigma_x^2}+D_{KL}+\text{cte.}")
c.para("Multiplicar por la constante positiva $2\\sigma_x^2$ no cambia el minimizador, así que $-\\mathrm{ELBO}\\propto\\Vert x-\\hat{x}\\Vert^2+2\\sigma_x^2D_{KL}$. "
       "Comparando con $\\mathcal{L}_\\beta=\\Vert x-\\hat{x}\\Vert^2+\\beta D_{KL}$ se obtiene $2\\sigma_x^2=\\beta$, es decir, **$\\sigma_x^2=\\beta/2$**.")
c.para(f"**Por qué salen borrosas.** Con $\\beta=1$ el modelo supone $\\sigma_x=0.71$ sobre píxeles en $[0,1]$. Ese \"ruido\" es mucho mayor que el contraste típico "
       f"entre píxeles vecinos de una imagen real ({s['real']:.3f}), así que a la pérdida le sale más barato tratar bordes, estampados y costuras como ruido "
       f"que codificarlos en $z$, donde cada nat cuesta $\\beta$. Lo que el decoder produce es $\\hat{{x}}=\\mathbb{{E}}[x\\mid z]$, el promedio de todas las imágenes que caen cerca del mismo $z$, "
       f"y promediar imágenes con bordes en posiciones distintas los difumina. Por eso las muestras tienen solo ≈ 49–58% de la nitidez real "
       f"({s['beta=0.1']:.4f}, {s['beta=1.0']:.4f} y {s['beta=10.0']:.4f}). La más borrosa es $\\beta=10$: con $\\sigma_x=2.24$, el ruido supuesto supera todo el rango de los píxeles. "
       f"$\\beta=0.1$ ($\\sigma_x=0.22$) no queda más nítida que $\\beta=1$ por dos razones: con latent_dim = 2 el cuello de botella ya limita los detalles que caben en $z$, "
       "y sus muestras del prior caen a menudo en los huecos del latente descritos en (c), donde el decoder entrega siluetas tenues. "
       "Además, se muestra la media $\\hat{x}$ y no una muestra $x\\sim\\mathcal{N}(\\hat{x},\\sigma_x^2I)$, que añadiría ruido blanco de desviación 0.71 y se vería aún peor.")

# ================================================================ Task 2.1
r23 = R["t2_3"]
c = C(F("Calcule el calendario lineal")._p.getnext())   # párrafo con la fórmula de x_t
c.para("Las imágenes se escalan a $[-1,1]$ con $x\\mapsto2x-1$. El calendario se implementa a mano en float64, con betas[t−1] $=\\beta_t$ para $t=1,\\dots,T$:")
c.eq(r"\beta_t=\beta_{start}+\frac{t-1}{T-1}\left(\beta_{end}-\beta_{start}\right),\qquad \alpha_t=1-\beta_t,\qquad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s")
c.para(f"Así $\\beta_1=10^{{-4}}$, $\\beta_T=0.02$ y $\\bar{{\\alpha}}_T={r23['ab_T']:.4e}$ (con torch.cumprod). La función q_sample(x0, t, eps) toma $\\bar{{\\alpha}}_t$ = alpha_bar[t − 1] para cada elemento del lote, "
       "le aplica .view(−1, 1, 1, 1) para hacer broadcasting y devuelve $x_t=\\sqrt{\\bar{\\alpha}_t}\\,x_0+\\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon$.")

# ================================================================ Task 2.2
r = R["t2_2"]
c = C(F("Fije una imagen"))
c.para("Se fijó $x_0$ = imagen 0 del conjunto de entrenamiento (Ankle boot, escalada a $[-1,1]$). Se crearon 5 000 copias (float64) y se aplicó 300 veces "
       "$x_s=\\sqrt{\\alpha_s}\\,x_{s-1}+\\sqrt{\\beta_s}\\,\\epsilon_s$, usando torch.randn_like en cada paso para que el ruido sea independiente por paso, copia y píxel.")
c = C(F("Calcule, por píxel, la media"))
c.para(f"Valores teóricos: $\\bar{{\\alpha}}_{{300}}={r['ab300']:.6f}$, $\\sqrt{{\\bar{{\\alpha}}_{{300}}}}={math.sqrt(r['ab300']):.6f}$ y $1-\\bar{{\\alpha}}_{{300}}={r['var_th']:.6f}$.")
c.table(["Cantidad", "Empírico (5 000 copias)", "Teórico"],
        [["Error máximo absoluto de la media (784 píxeles)", f"{r['max_err_mean']:.5f}", "0"],
         ["Error absoluto medio de la media", f"{r['mean_err_mean']:.5f}", "0"],
         ["Varianza promedio por píxel", f"{r['var_emp_mean']:.6f}", f"{r['var_th']:.6f}"],
         ["Rango de la varianza por píxel", f"[{r['var_emp_min']:.4f}, {r['var_emp_max']:.4f}]", f"{r['var_th']:.4f}"]])
c.image("t2_2_verificacion.png", 6.2, "Figura 8. Media teórica √ᾱ₃₀₀·x₀, media empírica, error normalizado por el error estándar y varianza empírica por píxel.")
c.para(f"La varianza empírica promedio ({r['var_emp_mean']:.6f}) coincide con $1-\\bar{{\\alpha}}_{{300}}$ con una diferencia de {abs(r['var_emp_mean']-r['var_th']):.1e}, "
       "y la media empírica reproduce la imagen original escalada por $\\sqrt{\\bar{\\alpha}_{300}}\\approx0.63$. El proceso paso a paso produce la misma distribución que la fórmula cerrada.")
se_var_avg = r["se_var"] / math.sqrt(784)
c = C(F("Indique si el error de la media es compatible"))
c.para(f"Sí es compatible. En cada píxel, la media de $n=5000$ copias tiene error estándar $SE=\\sqrt{{(1-\\bar{{\\alpha}}_{{300}})/n}}=\\sqrt{{{r['var_th']:.4f}/5000}}={r['se_mean']:.5f}$. "
       f"El error máximo, {r['max_err_mean']:.5f}, equivale a {r['max_over_se']:.2f} SE, pero es el **máximo sobre 784 píxeles** independientes. "
       f"La mediana teórica de $\\max|Z|$ para 784 normales estándar es $\\Phi^{{-1}}\\left((0.5^{{1/784}}+1)/2\\right)\\approx{r['expected_max_z']:.2f}$, es decir ≈ {r['expected_max_z']*r['se_mean']:.4f}, "
       "así que el valor observado es justamente el típico.")
c.para(f"Otras comprobaciones coinciden: el error absoluto medio ({r['mean_err_mean']:.5f}) frente a $SE\\sqrt{{2/\\pi}}={r['se_mean']*math.sqrt(2/math.pi):.5f}$, "
       f"la fracción de píxeles con |error| < 1.96 SE ({100*r['frac_within_196']:.2f}%, esperado 95%) y la desviación estándar de los errores normalizados ({r['z_std']:.3f}, esperado 1). "
       f"Para la varianza, el error estándar por píxel es $(1-\\bar{{\\alpha}})\\sqrt{{2/(n-1)}}={r['se_var']:.4f}$, coherente con el rango observado, y el del promedio sobre 784 píxeles es ≈ {se_var_avg:.5f}, "
       f"muy por encima de la diferencia observada de {abs(r['var_emp_mean']-r['var_th']):.1e}.")

# ================================================================ Task 2.3
c = C(F("Visualice  para"))
c.image("t2_3a_forward.png", 6.0, "Figura 9. x_t para Trouser, Sandal y Bag (mismo ε para todos los t de una fila).")
c.para("En $t=1$ la imagen es idéntica a $x_0$ ($\\mathrm{SNR}=9999$). En $t=50$ y $t=100$ aparece un granulado, pero la prenda se reconoce perfectamente. "
       "En $t=250$ ($\\mathrm{SNR}\\approx1.1$) apenas se intuye la silueta: el pantalón y el bolso todavía se adivinan, pero la sandalia, hecha de trazos finos, prácticamente desapareció. "
       "Desde $t=500$ las tres imágenes son indistinguibles de ruido gaussiano.")
c = C(F("Grafique  y la relación señal"))
c.image("t2_3b_alphabar_snr.png", 6.0, "Figura 10. ᾱ_t y SNR(t) (escala logarítmica) del calendario lineal.")
c.para(f"$\\bar{{\\alpha}}_t$ baja en forma de sigmoide y queda prácticamente en 0 después de $t\\approx700$. El log-SNR cae muy rápido al inicio (de $10^4$ a $10^2$ en solo 27 pasos) "
       f"y luego desciende casi linealmente hasta $\\mathrm{{SNR}}(T)={r23['snr_T']:.2e}$.")
c = C(F("Reporte el primer  con"))
c.para(f"Primer $t$ con $\\mathrm{{SNR}}(t)<1$: **t = {r23['t_snr1']}** ($\\mathrm{{SNR}}=0.9905$; en $t=259$ vale 1.0010). "
       f"Primer $t$ con $\\bar{{\\alpha}}_t<0.01$: **t = {r23['t_ab001']}** ($\\bar{{\\alpha}}=0.00999$). "
       "Es decir, desde $t=674$ la señal aporta menos del 1% de la varianza: casi un tercio de los pasos ocurre con $x_t$ prácticamente igual a ruido puro.")

# ================================================================ Task 2.4
r4 = R["t2_4"]
c = C(F("Diseñe un segundo calendario propio"))
c.para("**Calendario coseno truncado.** Parte del calendario coseno de Nichol & Dhariwal (2021), \"Improved Denoising Diffusion Probabilistic Models\", ICML:")
c.eq(r"\bar{\alpha}_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\qquad s=0.008")
c.para("En el original $\\bar{\\alpha}_T=0$ exactamente, lo que obliga a $\\beta_T=1$, y los autores tienen que recortar $\\beta_t\\le0.999$. "
       "**Modificación:** el argumento del coseno se multiplica por un factor $\\tau<1$, y $\\tau$ se elige por bisección para que $\\bar{\\alpha}_T=10^{-4}$:")
c.eq(r"f_\tau(t)=\cos^2\left(\tau\cdot\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\qquad \beta_t=1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}")
c.para(f"Resulta $\\tau={r4['tau']:.6f}$. Con este valor $\\beta_t\\in[{r4['beta_c_min']*1e5:.2f}\\times10^{{-5}},\\,{r4['beta_c_max']:.4f}]\\subset(0,1)$ sin necesidad de recortes, "
       "$\\bar{\\alpha}_T=10^{-4}<10^{-3}$, y el log-SNR conserva la forma casi lineal del coseno en la zona media.")
c.image("t2_4_calendarios.png", 6.5, "Figura 11. SNR(t), ᾱ_t y β_t del calendario lineal y del coseno truncado.")
c.table(["Calendario", "ᾱ_T", "Primer t con SNR(t) < 1"],
        [["Lineal (β de 10⁻⁴ a 0.02)", f"{r4['ab_T_lin']:.3e}", str(r4["t_snr1_lin"])],
         [f"Coseno truncado (τ = {r4['tau']:.4f})", f"{r4['ab_T_cos']:.3e}", str(r4["t_snr1_cos"])]])
c.para("El coseno destruye la señal mucho más despacio. Cruza $\\mathrm{SNR}=1$ justo a la mitad del proceso ($t=500$, contra $t=260$ en el lineal) "
       "y solo baja de $10^{-2}$ cerca del final ($t\\approx943$), donde $\\beta_t$ sube hasta 0.25 para alcanzar $\\bar{\\alpha}_T=10^{-4}$.")

# ================================================================ Task 2.5 (a)
a = R["t2_5a"]
c = C(F("Calcule a mano la primera suma"))
c.para("Con $\\beta_t=\\beta_1+(t-1)\\Delta$, $\\Delta=(\\beta_T-\\beta_1)/(T-1)$, $\\beta_1=10^{-4}$, $\\beta_T=0.02$ y $T=1000$:")
c.para("**Primera suma (aritmética):**")
c.eq(r"\sum_{t=1}^{T}\beta_t=T\cdot\frac{\beta_1+\beta_T}{2}=1000\cdot\frac{0.0001+0.02}{2}=1000\cdot0.01005=10.05")
c.para("**Segunda suma (aproximación integral):** los $\\beta_t$ están equiespaciados en $[\\beta_1,\\beta_T]$ con separación ≈ $(\\beta_T-\\beta_1)/T$, así que")
c.eq(r"\sum_{t=1}^{T}\beta_t^2\approx\frac{T}{\beta_T-\beta_1}\int_{\beta_1}^{\beta_T}b^2\,db=\frac{T\left(\beta_T^3-\beta_1^3\right)}{3\left(\beta_T-\beta_1\right)}=\frac{T\left(\beta_T^2+\beta_T\beta_1+\beta_1^2\right)}{3}")
c.eq(r"=\frac{1000\left(4\times10^{-4}+2\times10^{-6}+10^{-8}\right)}{3}=\frac{1000\cdot4.0201\times10^{-4}}{3}\approx0.13400")
c.para("**Resultado:**")
c.eq(r"\log\bar{\alpha}_T\approx-10.05-\frac{1}{2}(0.13400)=-10.11700\quad\Rightarrow\quad\bar{\alpha}_T\approx e^{-10.117}\approx4.04\times10^{-5}")
c.table(["Cantidad", "A mano", "Numérico (código)"],
        [["Σ β_t", f"{a['S1_hand']:.6f}", f"{a['S1_num']:.6f}"],
         ["Σ β_t²", f"{a['S2_hand']:.6f}", f"{a['S2_num']:.6f}"],
         ["log ᾱ_T", f"{a['log_approx']:.6f}", f"{a['log_exact']:.6f}"],
         ["ᾱ_T", f"{a['ab_approx']:.4e}", f"{a['ab_code']:.4e}"]])
c.para(f"El error relativo en $\\bar{{\\alpha}}_T$ es solo {100*abs(a['ab_approx']-a['ab_code'])/a['ab_code']:.3f}%. La primera suma es exacta. "
       f"La aproximación integral de $\\sum\\beta_t^2$ se queda corta por {a['S2_num']-a['S2_hand']:.1e}, y el resto de la diferencia en $\\log\\bar{{\\alpha}}_T$ lo explica el término omitido "
       f"$-\\tfrac13\\sum\\beta_t^3={a['third_term']:.5f}$: −10.11700 − 0.00067 − 0.00003 ≈ −10.11771, igual al valor exacto.")

# ================================================================ Task 2.5 (b)
b5 = R["t2_5b"]
c = C(F("Cuente, para cada calendario"))
c.table(["Calendario", "SNR > 100 (x_t ≈ x₀)", "SNR ∈ [0.01, 100]", "SNR < 0.01 (x_t ≈ ε)", "Rango de t útil"],
        [["Lineal", b5["lin"]["alto (>100)"], f"**{b5['lin']['útil [0.01,100]']}**", b5["lin"]["bajo (<0.01)"], f"{b5['range_lin'][0]}–{b5['range_lin'][1]}"],
         ["Coseno truncado", b5["cos"]["alto (>100)"], f"**{b5['cos']['útil [0.01,100]']}**", b5["cos"]["bajo (<0.01)"], f"{b5['range_cos'][0]}–{b5['range_cos'][1]}"]])
c.para("**Régimen $\\mathrm{SNR}\\ll0.01$ ($x_t\\approx\\epsilon$):** predecir $\\epsilon$ es trivial, porque $\\epsilon=(x_t-\\sqrt{\\bar{\\alpha}_t}x_0)/\\sqrt{1-\\bar{\\alpha}_t}\\approx x_t$ y basta con que la red copie su entrada. "
       "El error mínimo alcanzable es casi 0, el gradiente es casi nulo y el paso no enseña nada sobre la distribución de las prendas: es una iteración desperdiciada.")
c.para("**Régimen $\\mathrm{SNR}\\gg100$ ($x_t\\approx x_0$):** el ruido es diminuto frente a la imagen. Predecir $\\epsilon$ es difícil, porque hay que separar una perturbación "
       "pequeñísima de los detalles finos de la prenda (es donde la pérdida MSE en $\\epsilon$ es más alta). Pero ese error pesa poco en la imagen: el error en $x_0$ es "
       "el error en $\\epsilon$ dividido por $\\sqrt{\\mathrm{SNR}}$ (menos de 0.1). Estos pasos solo pulen detalles imperceptibles y aportan poco a la calidad de las muestras.")
c.para("**Régimen intermedio:** $x_t$ tiene a la vez estructura parcial y mucho ruido. Para predecir $\\epsilon$ la red tiene que \"saber\" cómo se ven las prendas "
       "(qué parte de $x_t$ es silueta y qué parte es ruido). Es aquí donde aprende la estructura de los datos, y es el rango que más aprovecha el muestreo.")
c.para(f"Como $t$ se sortea uniformemente, la fracción de iteraciones útiles es la fracción de pasos en el rango intermedio: **{b5['lin']['útil [0.01,100]']/10:.1f}%** con el lineal "
       f"frente a **{b5['cos']['útil [0.01,100]']/10:.1f}%** con el coseno truncado. El lineal gasta {b5['lin']['bajo (<0.01)']/10:.1f}% de las iteraciones "
       f"($t>{b5['range_lin'][1]}$) en ruido casi puro, contra {b5['cos']['bajo (<0.01)']/10:.1f}% del coseno. **El calendario coseno truncado es el más eficiente.** "
       "Predicción para la Task 3.5(d): con el calendario lineal, la pérdida en $\\epsilon$ debería ser máxima en el intervalo $t\\in[1,100]$ y casi nula para $t>700$.")

# ================================================================ Task 2.5 (c)
v = R["t2_5c"]
c = C(F("Demuestre que si  es la varianza global"))
c.para("**Demostración.** Sea $u$ un píxel elegido al azar (imagen y posición), con $a=\\sqrt{\\bar{\\alpha}_t}$ y $b=\\sqrt{1-\\bar{\\alpha}_t}$. Entonces $x_t(u)=a\\,x_0(u)+b\\,\\epsilon(u)$, "
       "donde $\\epsilon$ es independiente de $x_0$, con $\\mathbb{E}[\\epsilon]=0$ y $\\mathrm{Var}(\\epsilon)=1$ en cada píxel. Así")
c.eq(r"\mathrm{Var}(x_t)=a^2\,\mathrm{Var}(x_0)+b^2\,\mathrm{Var}(\epsilon)+2ab\,\mathrm{Cov}(x_0,\epsilon)=\bar{\alpha}_t\,\mathrm{Var}(x_0)+(1-\bar{\alpha}_t)")
c.para("La covarianza es 0 por la independencia entre el ruido y los datos. La media también se transforma: $\\mathbb{E}[x_t]=\\sqrt{\\bar{\\alpha}_t}\\,\\mathbb{E}[x_0]$.")
c.para(f"**Verificación numérica** sobre las 55 000 imágenes de entrenamiento en $[-1,1]$: $\\mathrm{{Var}}(x_0)={v['var_x0']:.5f}$ (media global {v['mean_x0']:.4f}).")
c.table(["t", "ᾱ_t", "Var(x_t) empírica", "Predicción ᾱ_t·Var(x₀) + (1 − ᾱ_t)", "Media empírica de x_t"],
        [[rw["t"], f"{rw['alpha_bar']:.5f}", f"{rw['var_emp']:.5f}", f"{rw['var_pred']:.5f}", f"{rw['mean_emp']:+.4f}"] for rw in v["rows"]])
xT = v["rows"][-1]
c.para(f"Las predicciones coinciden con los valores empíricos hasta la cuarta cifra decimal. **$x_T$ tiene varianza ≈ 1** ({xT['var_emp']:.5f}; teórico {xT['var_pred']:.5f}) "
       f"y media ≈ 0 ({xT['mean_emp']:+.4f}): como $\\bar{{\\alpha}}_T\\approx4\\times10^{{-5}}$, la señal residual $\\sqrt{{\\bar{{\\alpha}}_T}}\\,x_0$ tiene amplitud ≈ 0.006. "
       "Es necesario que sea 1 porque la generación empieza en $x_T\\sim\\mathcal{N}(0,I)$, y la red solo aprendió a quitar ruido de los $x_T$ que produce el proceso forward. "
       "Si $q(x_T)$ no coincidiera con $\\mathcal{N}(0,I)$ (por ejemplo, con varianza 0.5 o media −0.43 como $x_0$), el muestreo arrancaría de una distribución que la red nunca vio en $t=T$ "
       "y ese desajuste se propagaría por toda la cadena inversa. La forma que preserva la varianza (coeficientes $\\sqrt{\\alpha_t}$ y $\\sqrt{\\beta_t}$) garantiza que la varianza tienda a 1 "
       "sin importar $\\mathrm{Var}(x_0)$, y un $\\bar{\\alpha}_T$ pequeño garantiza que se borre la información de $x_0$.")

doc.save(DST)
print("guardado", DST)
