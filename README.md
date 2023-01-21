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


