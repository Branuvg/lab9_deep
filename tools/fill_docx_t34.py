"""Inserta las respuestas de las Tasks 3.4 y 3.5(a), (b), (c) en el docx del laboratorio.
Uso: python fill_docx_t34.py <entrada.docx> <destino.docx> <raíz del repo>
<entrada.docx> puede ser la plantilla o el documento que ya generó fill_docx.py (las anclas son los enunciados).
Texto: **negrita**, $latex$ en línea; c.eq(latex) para ecuaciones centradas. Números: results/results_t3_4_t3_5.json."""
import json, sys
import numpy as np
import docx
from docx_helpers import Cursor, find

SRC, DST, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(f"{ROOT}/results/results_t3_4_t3_5.json", encoding="utf8"))
doc = docx.Document(SRC)
C = lambda anchor: Cursor(doc, anchor, f"{ROOT}/figures")
F = lambda prefix: find(doc, prefix)
CL = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
pc = lambda x: f"{100*x:.1f}%"

c4a, c4b, c4c, c4d = R["t3_4a"], R["t3_4b"], R["t3_4c"], R["t3_4d"]
a5, b5, c5 = R["t3_5a"], R["t3_5b"], R["t3_5c"]
fid = {w: c4c[w]["fidelity"] for w in ["1", "3", "7", "real"]}
div = {w: c4c[w]["diversity"] for w in ["1", "3", "7", "real"]}

# ================================================================ Task 3.4 (a)
c = C(F("Implemente la predicción guiada")._p.getnext())      # párrafo de la ecuación
c.para("La predicción guiada está en la función make_guided_eps_fn del notebook. Recibe el modelo y $w$ y devuelve una función $\\epsilon(x_t,t,c)$ "
       "que se pasa sin cambios a ancestral_sample de la Task 3.3, así que el bucle de muestreo es el mismo y solo cambia la predicción de ruido:")
c.eq(r"\tilde\epsilon=\epsilon_\theta(x_t,t,\varnothing)+w\left(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\right),\qquad x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\tilde\epsilon\right)+\sigma_t z")
c.para("Para $w>1$ se evalúa la red con el lote duplicado (una mitad con la clase pedida y otra con la condición nula 10) en una sola pasada, es decir, 2 evaluaciones de la red "
       "por paso. Para $w=1$ la fórmula se reduce a $\\epsilon_\\theta(x_t,t,c)$ y se evita la segunda evaluación (1 evaluación por paso). "
       f"Comprobación numérica contra la fórmula con dos evaluaciones separadas: error máximo {c4a['max_abs_diff']['1']:.1e} con $w=1$, "
       f"{c4a['max_abs_diff']['3']:.1e} con $w=3$ y {c4a['max_abs_diff']['7']:.1e} con $w=7$ (redondeo de punto flotante amplificado por $w$).")

# ================================================================ Task 3.4 (b)
c = C(F("Genere 50 imágenes por clase"))
c.para("Se generaron 5 lotes de 100 imágenes con 10 por clase cada uno, es decir, 50 imágenes por clase y 500 por cada $w$. Cada lote $k$ usa la semilla $42+k$ y es el mismo "
       "para los tres valores de $w$: misma $x_T$ y mismo ruido $z$, de modo que las diferencias entre $w$ se deben solo a la guía. Las imágenes (en $[-1,1]$, recortadas) quedaron en "
       "results/samples_w1.pt, samples_w3.pt y samples_w7.pt, junto con sus clases y los tiempos.")

# ================================================================ Task 3.4 (c)
c = C(F("Entrene un clasificador sencillo"))
c.para(f"**Clasificador.** CNN de dos convoluciones (1→32→64, con ReLU y max-pooling) y dos capas lineales (3136→128→10), entrenada con Adam (lr = $10^{{-3}}$, lotes de 128) "
       f"solo con las 55 000 imágenes reales de entrenamiento. Alcanzó **{100*c4c['clf_val_acc']:.2f}%** de exactitud en las 5 000 de validación después de {c4c['clf_epochs']} épocas "
       "(requisito: al menos 88%). Se guardó en checkpoints/classifier.pt.")
c.para("**Fidelidad:** fracción de las imágenes generadas que el clasificador asigna a la clase pedida. **Diversidad:** distancia euclidiana promedio, sobre los 784 píxeles en $[-1,1]$, "
       "entre los $50\\cdot49/2=1225$ pares de imágenes de una misma clase, promediada sobre las 10 clases. Como referencia se usaron las primeras 50 imágenes reales de cada clase en validación: "
       f"el clasificador acierta {pc(fid['real'])} de ellas y su diversidad es {div['real']:.3f}.")

# ================================================================ Task 3.4 (d)
c = C(F("Presente una tabla con"))
rows = [[w, pc(fid[w]), f"{div[w]:.3f}", f"{c4b[w]['time_per_100_s']:.1f} s", c4b[w]["n_net_evals_per_image"]] for w in ["1", "3", "7"]]
rows.append(["Reales (50 por clase)", pc(fid["real"]), f"{div['real']:.3f}", "—", "—"])
c.table(["w", "Fidelidad global", "Diversidad global", "Tiempo por 100 imágenes", "Evaluaciones de red por imagen"], rows,
        "Tabla. Fidelidad, diversidad y tiempo de generación (GPU, lotes de 100 imágenes, 1 000 pasos).")
c.image("t3_4d_muestras_clase.png", 6.5,
        f"Figura. 10 muestras de la clase {c4d['class_in_figure']} para w = 1 (arriba), 3 y 7 (abajo); cada columna parte de la misma x_T.")
c.para(f"La clase de la figura es {c4d['class_in_figure']}, la de mayor caída de diversidad entre $w=1$ y $w=7$ (se analiza en 3.5(a)). "
       f"El tiempo con $w=3$ y $w=7$ es casi el doble que con $w=1$ ({c4b['3']['time_per_100_s']:.1f} s y {c4b['7']['time_per_100_s']:.1f} s frente a {c4b['1']['time_per_100_s']:.1f} s) "
       "porque cada paso necesita dos evaluaciones de la red; $w=3$ y $w=7$ cuestan lo mismo entre sí.")

# ================================================================ Task 3.5 (a)
rows_a, sup = a5["rows"], {s["class"]: s for s in a5["support"]}
drops = np.array([r["drop_pct"] for r in rows_a])
k = int(np.argmax(drops)); rk = rows_a[k]; sk = sup[rk["class"]]
r_div = float(np.corrcoef(drops, [r["div_real"] for r in rows_a])[0, 1])
r_fid = float(np.corrcoef(drops, [r["fid"][0] for r in rows_a])[0, 1])
bag = rows_a[8]; tsh = rows_a[0]
c = C(F("Describa cómo cambian"))
c.table(["Clase", "Div. real", "Div. w=1", "Div. w=3", "Div. w=7", "Caída 1→7", "Fid. w=1", "Fid. w=3", "Fid. w=7"],
        [[r["class"], f"{r['div_real']:.2f}", f"{r['div'][0]:.2f}", f"{r['div'][1]:.2f}", f"{r['div'][2]:.2f}", f"{r['drop_pct']:.1f}%",
          pc(r["fid"][0]), pc(r["fid"][1]), pc(r["fid"][2])] for r in rows_a])
c.image("t3_5a_diversidad_fidelidad_por_clase.png", 6.3, "Figura. Diversidad y fidelidad por clase para w = 1, 3 y 7 (línea negra: global).")
c.para(f"**Fidelidad.** Sube de {pc(fid['1'])} con $w=1$ a {pc(fid['3'])} con $w=3$ y baja a {pc(fid['7'])} con $w=7$. "
       f"La baja de $w=7$ viene de una sola clase: Bag cae a {pc(bag['fid'][2])}, mientras que las otras nueve clases llegan al 100%. "
       f"Con $w=7$ las bolsas salen fragmentadas o casi vacías (la intensidad media de la imagen baja de {sup['Bag']['ink_w1']:.3f} a {sup['Bag']['ink_w7']:.3f}) y el clasificador ya no las reconoce. "
       "Es decir, la guía mejora la fidelidad hasta un punto y después, con $w$ demasiado grande, empieza a producir imágenes fuera de la distribución.")
c.para(f"**Diversidad.** Baja de forma continua: {div['1']:.2f} con $w=1$, {div['3']:.2f} con $w=3$ y {div['7']:.2f} con $w=7$ ({100*(div['1']-div['7'])/div['1']:.0f}% menos entre $w=1$ y $w=7$), "
       f"frente a {div['real']:.2f} de las imágenes reales. Incluso con $w=1$ el modelo ya es menos diverso que los datos reales. "
       f"Una excepción aparente es T-shirt/top ({tsh['div'][1]:.2f} con $w=3$ y {tsh['div'][2]:.2f} con $w=7$): al ver las imágenes, el aumento viene de franjas y artefactos horizontales que aparecen con $w=7$, "
       "que la distancia en píxeles cuenta como variedad aunque no lo sea.")
c.para(f"**Clase más afectada: {rk['class']}**, que baja de {rk['div'][0]:.2f} a {rk['div'][2]:.2f} ({rk['drop_pct']:.1f}%). "
       f"Hipótesis: es una clase con un solo modo muy marcado (dos piernas rectas en una silueta casi fija) y sin vecinos con los que se confunda, así que la guía no tiene ambigüedad que resolver "
       f"y lo único que hace es empujar todas las muestras hacia ese patrón central, eliminando la poca variedad que había (ancho de las piernas, separación y brillo). Los datos lo respaldan:")
c.para(f"(1) Ya con imágenes reales es una de las clases menos diversas ({rk['div_real']:.2f}; solo Sneaker tiene menos), y entre las 10 clases las que más caen son las que parten con menos diversidad real "
       f"(correlación entre la caída y la diversidad real: $r={r_div:+.2f}$). "
       f"(2) Con $w=1$ ya es de las más fáciles de reconocer ({pc(rk['fid'][0])}), y las clases que ya se reconocen bien son las que más pierden ($r={r_fid:+.2f}$ entre caída y fidelidad con $w=1$), "
       f"mientras que Shirt, la más ambigua ({pc(rows_a[6]['fid'][0])}), solo cae {rows_a[6]['drop_pct']:.1f}%. "
       f"(3) La distancia entre siluetas binarias baja de {sk['mask_div_w1']:.2f} a {sk['mask_div_w7']:.2f} ({100*(sk['mask_div_w1']-sk['mask_div_w7'])/sk['mask_div_w1']:.0f}%) y la desviación estándar del brillo medio "
       f"entre imágenes baja de {sk['bright_std_w1']:.3f} a {sk['bright_std_w7']:.3f} ({100*(sk['bright_std_w1']-sk['bright_std_w7'])/sk['bright_std_w1']:.0f}%): "
       "se pierden tanto la forma como el brillo, y en la figura las 10 muestras con $w=7$ ya son casi la misma prenda.")

# ================================================================ Task 3.5 (b)
cm = np.array(b5["confusion_matrix_w1"]); A, B = [CL.index(n) for n in b5["pair"]]
off = cm.copy(); np.fill_diagonal(off, 0)
g_ab, g_ba = b5["gen_confusions"]; (r_ab, r_ba), (gr_ab, gr_ba) = b5["real_val_rate"], b5["gen_rate"]
c = C(F("Calcule la matriz de confusión"))
c.image("t3_5b_matriz_confusion.png", 6.5,
        f"Figura. Izquierda: matriz de confusión del clasificador sobre las 500 imágenes generadas con w = 1 (filas: clase pedida). Derecha: imágenes del par {b5['pair'][0]} / {b5['pair'][1]}.")
c.para(f"**Par más confundido: {b5['pair'][0]} y {b5['pair'][1]}.** {g_ab} de las 50 imágenes pedidas como {b5['pair'][0]} se clasificaron como {b5['pair'][1]} y "
       f"{g_ba} de las 50 pedidas como {b5['pair'][1]} se clasificaron como {b5['pair'][0]} ({g_ab+g_ba} de 100). "
       f"Le siguen Shirt → Pullover ({off[6, 2]}), Coat → Pullover ({off[4, 2]}), Pullover → Coat ({off[2, 4]}) y T-shirt/top → Shirt ({off[0, 6]}): todas son prendas de tronco con manga. "
       f"Las clases de calzado, Trouser y Dress casi no se confunden (recall de {pc(min(b5['recall_by_class'][i] for i in [1, 3, 5, 7, 9]))} o más).")
c.para(f"**¿De quién es el defecto?** Para separarlo se midió la misma confusión con imágenes reales de validación: {b5['pair'][0]} → {b5['pair'][1]} = {100*r_ab:.1f}% "
       f"y {b5['pair'][1]} → {b5['pair'][0]} = {100*r_ba:.1f}%, contra {100*gr_ab:.1f}% y {100*gr_ba:.1f}% en las generadas. "
       f"En las reales la confusión ya existe, sobre todo de Shirt hacia Coat (el clasificador solo llega a 90.5% en validación), así que una parte es de las propias clases y del clasificador: "
       "Coat y Shirt comparten silueta, manga larga y abertura frontal, y en las imágenes reales de la figura (filas inferiores) hay camisas con aspecto de abrigo y abrigos sin textura distintiva. "
       f"Pero el exceso en las generadas (unos {100*(gr_ab-r_ab):.0f} y {100*(gr_ba-r_ba):.0f} puntos más) sí es atribuible al generador: con $w=1$ produce prendas con manga larga y textura "
       f"poco definida, a medio camino entre las dos clases (filas superiores de la figura). La evidencia es que con $w=3$ y $w=7$, donde la guía obliga a rasgos más propios de la clase, "
       f"la fidelidad de Coat sube de {pc(rows_a[4]['fid'][0])} a {pc(rows_a[4]['fid'][1])} y la de Shirt de {pc(rows_a[6]['fid'][0])} a {pc(rows_a[6]['fid'][1])}. "
       "**Conclusión:** la confusión es sobre todo una propiedad de las clases (Coat y Shirt son parecidas incluso en datos reales), amplificada por un generador sin guía que no marca "
       "lo suficiente los rasgos de cada una; el clasificador contribuye poco, porque falla poco en las demás clases.")

# ================================================================ Task 3.5 (c)
rms = c5["guidance_rms_by_interval"]; ks = list(rms)
c = C(F("Demuestre que la predicción guiada"))
c.para("**Demostración.** Sea $k_t=1/\\sqrt{1-\\bar\\alpha_t}$. Con la aproximación del enunciado, el score condicional y el no condicional son")
c.eq(r"\nabla_{x_t}\log p(x_t\mid c)\approx-k_t\,\epsilon_\theta(x_t,t,c),\qquad \nabla_{x_t}\log p(x_t)\approx-k_t\,\epsilon_\theta(x_t,t,\varnothing)")
c.para("Se multiplica la predicción guiada por $-k_t$:")
c.eq(r"-k_t\tilde\epsilon=-k_t\epsilon_\theta(x_t,t,\varnothing)+w\left(-k_t\epsilon_\theta(x_t,t,c)+k_t\epsilon_\theta(x_t,t,\varnothing)\right)\approx\nabla\log p(x_t)+w\left(\nabla\log p(x_t\mid c)-\nabla\log p(x_t)\right)")
c.para("Por Bayes, $p(x_t\\mid c)=p(c\\mid x_t)\\,p(x_t)/p(c)$. Como $p(c)$ no depende de $x_t$, al tomar logaritmo y gradiente:")
c.eq(r"\nabla\log p(x_t\mid c)=\nabla\log p(c\mid x_t)+\nabla\log p(x_t)\quad\Rightarrow\quad\nabla\log p(x_t\mid c)-\nabla\log p(x_t)=\nabla\log p(c\mid x_t)")
c.para("Sustituyendo en la expresión anterior:")
c.eq(r"-k_t\tilde\epsilon\approx\nabla\log p(x_t)+w\,\nabla\log p(c\mid x_t)=\nabla\log\left[p(x_t)\,p(c\mid x_t)^w\right]")
c.para("Entonces $\\tilde\\epsilon$ es, salvo el factor $-1/k_t$, el score de $\\tilde p(x\\mid c)\\propto p(x)\\,p(c\\mid x)^w$, y el muestreo con $\\tilde\\epsilon$ sigue ese score. "
       "Con $w=1$ el factor es $p(x)p(c\\mid x)\\propto p(x\\mid c)$ (el condicional puro) y con $w=0$ queda $p(x)$ (el modelo no condicional).")
c.para("**Diversidad.** El factor $p(c\\mid x)^w$ eleva a la potencia $w$ la probabilidad de la clase. Al crecer $w$, las regiones donde $p(c\\mid x)$ es casi 1 pesan cada vez más "
       "frente a las que tienen $p(c\\mid x)$ pequeña (bordes de la clase y zonas ambiguas con otras), así que $\\tilde p$ se concentra en las imágenes más típicas de la clase, "
       "igual que bajar la temperatura de una distribución. Las muestras se parecen más entre sí: la diversidad cae, como se midió en la 3.4 "
       f"({div['1']:.2f} → {div['3']:.2f} → {div['7']:.2f}). Con $w$ muy grande el término $p(c\\mid x)^w$ puede también sacar la muestra de la distribución de los datos, "
       f"que es lo que se vio con las bolsas a $w=7$ (fidelidad {pc(bag['fid'][2])}).")
c.para(f"**Tamaño de la corrección en nuestro modelo.** La guía añade $(w-1)(\\epsilon_c-\\epsilon_\\varnothing)$ a la predicción condicional. En 1 000 imágenes de validación el RMS por píxel de "
       f"$\\epsilon_c-\\epsilon_\\varnothing$ es {rms[ks[0]]:.3f} para $t\\in[1,100]$, {rms[ks[5]]:.3f} para $t\\in[501,600]$ y {rms[ks[9]]:.3f} para $t\\in[901,1000]$: "
       "la guía actúa sobre todo en los pasos con SNR intermedio, donde se decide la forma de la prenda, y casi no actúa cuando $x_t$ es ruido puro. "
       f"En norma por imagen, $\\Vert\\epsilon_c-\\epsilon_\\varnothing\\Vert$ es en promedio unas 6.5% de $\\Vert\\epsilon_c\\Vert$, pero con $w=7$ se multiplica por $w-1=6$.")

doc.save(DST)
print("guardado", DST)
