# DTMF Decoder (Goertzel Algorithm) 📞

Proyecto final para la asignatura Computación Científica que implementa un sistema de procesamiento digital de señales (DSP) **generación y decodificación de tonos telefónicos (DTMF)**.

Utiliza el **Algoritmo de Goertzel** para lograr una detección de frecuencias eficiente y robusta, superando en rendimiento a la FFT estándar para este caso de uso específico.

## 🚀 Características Clave

* **Algoritmo de Goertzel Optimizado:** Complejidad $O(N)$ vs $O(N \log N)$ de la FFT.
* **Robustez ante Ruido:** Implementa validación por *Ratio Señal-Ruido (SNR)* para evitar falsos positivos en entornos ruidosos.
* **Visualización:** Genera gráficas sincronizadas de oscilograma, espectrograma y matriz de tonos DTMF.
* **Modularidad:** Lógica separada en módulos reutilizables (`dtmf_tools.py`).

## 🛠️ Instalación

Clona el repositorio e instala las dependencias necesarias:

```bash
git clone https://github.com/Tadeo-AR26/DTMFsignaling.git
cd DTMFsignaling
pip install numpy matplotlib scipy