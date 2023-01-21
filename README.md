# Watershed (Segmentación)

Watershed es un algoritmo clásico utilizado para la segmentación y es especialmente útil cuando se *extraen objetos que se tocan* o *se superponen* en imágenes.

Al utilizar el algoritmo, se inicia con *marcadores definidos por el usuario*. Estos marcadores se pueden definir *manualmente* mediante apuntar y hacer clic, o podemos definirlos *automática* o *heurísticamente* utilizando métodos como umbrales y/o operaciones morfológicas.

Basándose en estos marcadores, el algoritmo trata los píxeles en nuestra imagen de entrada como elevación local. El método "inunda" los valles, comenzando desde los marcadores y moviéndose hacia afuera, hasta que los valles de diferentes marcadores se encuentran entre sí. Para obtener una segmentación precisa, los marcadores deben colocarse correctamente.

### Módulos requeridos

Los módulos requeridos son los siguientes:

- Scipy 

  ```
  pip install scipy
  ```

- Scikit-image

  ```
  pip install scikit-image
  ```

- OpenCV

  ```
  pip install opencv-python
  ```

- Imutils

  ```
  pip install imutils
  ```

### Procedimiento

Para el análisis utilizaremos la siguiente imagen de prueba:

<img src="https://github.com/carlosjulioph/Watershed-Segementation/blob/main/images/1.jpg" width="500">

- ### Pre-procesamiento

```
img = cv2.imread('1.jpg')
img = cv2.resize(img,None,fx=0.5, fy=0.5,
                        interpolation = cv2.INTER_LINEAR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
```

<img src="https://github.com/carlosjulioph/Watershed-Segementation/blob/main/images/thresh.png" width="500">

- ### Transformación de distancia

Ahora tenemos que tratar de separar los objetos que se tocan y tratar de crear un borde entre ellos. Una idea es crear un borde lo más lejos posible de los centros de los objetos superpuestos. Luego, usaremos una técnica llamada transformación de distancia. Es un operador que generalmente toma imágenes binarias como entradas y las intensidades de píxeles de los puntos dentro de las regiones en primer plano se reemplazan por su distancia al píxel más cercano con intensidad cero (píxel de fondo). Podemos usar la función `distance_transform_edt()` de la biblioteca scipy. Esta función calcula la transformada exacta de la distancia euclidiana.

```
dist = ndi.distance_transform_edt(thresh)
dist_visual = dist.copy()
#para visualizar a traves de cv2.imshow()
dist_visual = cv2.normalize(src=dist_visual, dst=None, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
```

<img src="https://github.com/carlosjulioph/Watershed-Segementation/blob/main/images/dist.png" width="500">

Ahora necesitamos encontrar las coordenadas de los picos (máximos locales) de las áreas blancas en la imagen. Para eso, usaremos la función `peak_local_max()` de la biblioteca de imágenes Scikit. Aplicaremos esta función a nuestra imagen y la salida nos dará los marcadores que se utilizarán en la función de watershed.

```
local_max = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)
```
El siguiente paso es etiquetar marcadores para la función de watershed. Para eso, usaremos la función `ndi.label()` de la biblioteca SciPy. La función etiquetará aleatoriamente todos los máximos locales calculados previamente con diferentes valores positivos a partir de 1. Entonces, en caso de que tengamos 10 objetos en la imagen cada uno de ellos estará etiquetado con un valor del 1 al 10.

```
markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
```

- ### Watershed

El paso final es aplicar la función `watershed()` de la biblioteca de imágenes Scikit. Como parámetros, necesitamos pasar nuestra imagen de transformación de distancia invertida y los marcadores que calculamos en la línea de código anterior. Dado que el algoritmo de watershed asume que nuestros marcadores representan mínimos locales, necesitamos invertir nuestra imagen de transformación de distancia. De esa manera, los píxeles claros representarán elevaciones altas, mientras que los píxeles oscuros representarán las elevaciones bajas para la transformación de watershed.

```
labels = watershed(-dist, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
```

Para visualizar el resultado de etiquetas, utilizamos las siguientes líneas de código,

```
from skimage import exposure
img_watershed  = exposure.rescale_intensity(labels, out_range=(0, 255))
img_watershed  = np.uint8(img_watershed )
img_watershed  = cv2.applyColorMap(img_watershed , cv2.COLORMAP_JET)
```
<img src="https://github.com/carlosjulioph/Watershed-Segementation/blob/main/images/watershed.png" width="500">

El último paso es simplemente recorrer los valores de etiqueta únicos y extraer cada uno de los objetos únicos:

```
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
```
<img src="https://github.com/carlosjulioph/Watershed-Segementation/blob/main/images/output.png" width="500">

