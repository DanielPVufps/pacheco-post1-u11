# Post-Contenido 1 — CUDA Benchmark CPU vs GPU

## Estudiante

**Nombre:** Daniel Pacheco Villamizar

**Asignatura:** Arquitectura de Computadores

---

## Objetivo

Implementar kernels CUDA para suma de vectores y multiplicación de matrices, gestionando explícitamente la memoria de GPU, comparando el rendimiento CPU vs GPU y documentando los resultados obtenidos.

---

## Entorno de Trabajo

### Sistema Operativo

Ubuntu 26.04 LTS sobre WSL2

### Compilador

```bash
gcc 15.2.0
```

### CUDA Toolkit

```bash
CUDA Toolkit 12.4
```

### GPU Detectada en Windows

```text
NVIDIA GeForce GTX 1650
CUDA Version 11.2
```

---

## Estructura del Proyecto

```text
pacheco-post1-u11/
│
├── README.md
├── capturas/
│
└── src/
    ├── vectorAdd.cu
    ├── vectorAdd_resultados.txt
    ├── matMul.cu
    └── matMul_resultados.txt
```

---

## Implementación 1: Vector Addition

Se implementó un kernel CUDA donde cada hilo calcula un elemento del vector resultado.

Características:

* Benchmark CPU
* Benchmark GPU Kernel
* Benchmark GPU Total
* Verificación de errores
* Manejo de errores CUDA
* Tamaños evaluados:

  * 1M elementos
  * 4M elementos
  * 16M elementos

### Resultados

| Tamaño | CPU (ms) | GPU Kernel (ms) | GPU Total (ms) |
| ------ | -------- | --------------- | -------------- |
| 1M     | N/D      | N/D             | N/D            |
| 4M     | N/D      | N/D             | N/D            |
| 16M    | N/D      | N/D             | N/D            |

---

## Implementación 2: Matrix Multiplication

Se implementaron dos versiones:

### Naive

Cada hilo calcula un elemento de la matriz resultado utilizando únicamente memoria global.

### Tiled Shared Memory

Se utilizó:

```cpp
#define TILE 16
```

para reducir accesos a memoria global mediante memoria compartida (shared memory).

Características:

* Multiplicación CPU
* Multiplicación GPU Naive
* Multiplicación GPU Tiled
* Shared Memory
* Benchmark con cudaEvent
* Cálculo de speedup
* Verificación de resultados

### Resultados

| Tamaño    | CPU (ms) | GPU Naive (ms) | GPU Tiled (ms) | Speedup |
| --------- | -------- | -------------- | -------------- | ------- |
| 512x512   | N/D      | N/D            | N/D            | N/D     |
| 1024x1024 | N/D      | N/D            | N/D            | N/D     |

---

## Análisis

La arquitectura GPU permite ejecutar miles de hilos en paralelo, lo que hace que operaciones altamente paralelizables como la suma de vectores y la multiplicación de matrices puedan ejecutarse significativamente más rápido que en CPU cuando el tamaño de los datos es suficientemente grande.

En multiplicación de matrices, el uso de Shared Memory reduce considerablemente el número de accesos a memoria global, permitiendo reutilizar datos cargados dentro de cada bloque CUDA. Esta técnica mejora el rendimiento respecto a una implementación naïve y constituye una de las optimizaciones más importantes en programación CUDA.

---

## Checkpoints

### Checkpoint 1

Compilación y ejecución de vectorAdd:

```bash
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd
```

### Checkpoint 2

Compilación y ejecución de matMul:

```bash
nvcc -O2 -o matMul src/matMul.cu
./matMul
```

### Checkpoint 3

Resultados y documentación almacenados en GitHub.

---

## Archivos Entregados

* README.md
* src/vectorAdd.cu
* src/matMul.cu
* src/vectorAdd_resultados.txt
* src/matMul_resultados.txt

---

## Conclusiones

La práctica permitió implementar y analizar algoritmos paralelos utilizando CUDA. Se desarrollaron kernels para suma de vectores y multiplicación de matrices utilizando memoria global y shared memory. Además, se aplicaron técnicas de medición de rendimiento mediante cudaEvent y se comparó el comportamiento de CPU y GPU para diferentes tamaños de problema.
