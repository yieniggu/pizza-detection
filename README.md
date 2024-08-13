### Instrucciones

Clonar el repositorio

```shell
git clone https://github.com/yieniggu/pizza-detection.git
cd pizza-detection
```

Para ejecutar sin ver ni guardar los resultados como video
```shell
python3 yolotrack_csv_only.py -v path/to/video
```

Para ejecutar guardando los resultados en video y visualizar en tiempo real
```shell
python3 yolotrack.py -v path/to/video
```

Librerias requeridas
```
ultralytics
opencv
supervision
imutils
```