import cv2
import numpy as np
import os
import random
import csv


# 1. Configuración Principal
NUM_IMAGENES = 3000
IMG_SIZE = 64
CARPETA_SALIDA = "dataset_imagenes"
ARCHIVO_CSV = "etiquetas.csv"

# Definimos nuestras opciones de Formas y Colores
# OpenCV usa formato BGR (Blue, Green, Red)
COLORES = {
    "rojo": (0, 0, 255),
    "verde": (0, 255, 0),
    "azul": (255, 0, 0)
}
FORMAS = ["cuadrado", "circulo", "rectangulo","triangulo"]

# 2. Preparación del Entorno
# Crear la carpeta principal si no existe
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

# Abrir el archivo CSV en modo escritura
# newline = '' para evitar saltos de linea extra
archivo_csv = open(ARCHIVO_CSV, mode='w', newline='')
escritor_csv = csv.writer(archivo_csv)

# Escribir la cabecera del CSV
escritor_csv.writerow(["nombre_archivo", "forma", "color"])

print(f"Iniciando la generación de {NUM_IMAGENES} imágenes...")

# 3. Bucle de Generación
for i in range(NUM_IMAGENES):
    # a) Elegir características al azar
    forma_elegida = random.choice(FORMAS)
    nombre_color = random.choice(list(COLORES.keys()))
    valor_bgr = COLORES[nombre_color]
    
    # b) Crear lienzo blanco (Alto, Ancho, 3 canales)
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    
    # c) Dibujar la forma asegurando posiciones y tamaños aleatorios
    if forma_elegida == "cuadrado":
        lado = random.randint(15, 30)
        x = random.randint(0, IMG_SIZE - lado)
        y = random.randint(0, IMG_SIZE - lado)
        cv2.rectangle(img, (x, y), (x + lado, y + lado), valor_bgr, -1)
        
    elif forma_elegida == "circulo":
        radio = random.randint(8, 15)
        x = random.randint(radio, IMG_SIZE - radio)
        y = random.randint(radio, IMG_SIZE - radio)
        cv2.circle(img, (x, y), radio, valor_bgr, -1)
        
    elif forma_elegida == "rectangulo":
        ancho = random.randint(20, 40)
        alto = random.randint(10, 20)
        x = random.randint(0, IMG_SIZE - ancho)
        y = random.randint(0, IMG_SIZE - alto)
        cv2.rectangle(img, (x, y), (x + ancho, y + alto), valor_bgr, -1)
        
    elif forma_elegida == "triangulo":
        # Para el triángulo, definimos una "caja" imaginaria (base y altura)
        base = random.randint(20, 35)
        altura = random.randint(20, 35)
        x = random.randint(0, IMG_SIZE - base)
        y = random.randint(0, IMG_SIZE - altura)
        
        # Calculamos los 3 vértices dentro de esa caja
        punto1 = [x + base // 2, y]            # Punta superior (mitad de la base, arriba)
        punto2 = [x, y + altura]               # Esquina inferior izquierda
        punto3 = [x + base, y + altura]        # Esquina inferior derecha
        
        # Convertimos los puntos al formato de matriz que exige OpenCV
        vertices = np.array([[punto1, punto2, punto3]], np.int32)
        
        # Dibujamos y rellenamos el polígono
        cv2.fillPoly(img, vertices, valor_bgr)
    
    # d) Guardar la imagen en el disco
    nombre_img = f"img_{i:04d}.png" # Formato img_0000.png, img_0001.png...
    ruta_img = os.path.join(CARPETA_SALIDA, nombre_img)
    cv2.imwrite(ruta_img, img)
    
    # e) Anotar en el registro CSV
    escritor_csv.writerow([nombre_img, forma_elegida, nombre_color])

# 4. Cierre
archivo_csv.close()
print(f"¡Proceso finalizado! Las imágenes están en '{CARPETA_SALIDA}' y el registro en '{ARCHIVO_CSV}'.")
