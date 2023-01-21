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
