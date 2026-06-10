# PREFIN — Plataforma Inteligente de Predicción y Prevención Financiera

PREFIN es una aplicación web desarrollada como Trabajo Fin de Grado para el análisis de finanzas personales. La plataforma permite cargar transacciones bancarias, analizarlas automáticamente, clasificar gastos por categorías, estimar el riesgo financiero del usuario mediante Machine Learning, simular escenarios futuros mediante un gemelo digital y calcular estrategias de micro-ahorro.

La aplicación puede utilizarse de tres formas:

1. Con **datos sintéticos**, generados automáticamente desde la propia aplicación.
2. Con un archivo **CSV o Excel** de transacciones.
3. Con conexión bancaria mediante **TrueLayer Sandbox**.

Para una primera prueba o evaluación del proyecto, se recomienda utilizar los **datos sintéticos**, ya que no requieren credenciales externas.

---

## 1. Funcionalidades principales

La aplicación incluye las siguientes secciones:

### Datos

Permite seleccionar la fuente de información financiera:

* generación de datos sintéticos;
* carga de archivos CSV o Excel;
* conexión con TrueLayer Sandbox mediante Open Banking.

### Dashboard

Muestra una visión general de la situación financiera del usuario:

* ingresos;
* gastos;
* ahorro;
* saldo;
* evolución mensual;
* últimas transacciones.

### Análisis financiero

Analiza el comportamiento financiero del usuario:

* gasto por categoría;
* evolución de ingresos y gastos;
* detección de anomalías;
* tendencias de consumo;
* distribución temporal de los gastos.

### Riesgo ML

Utiliza un modelo de Machine Learning para estimar el nivel de riesgo financiero del usuario.

El modelo clasifica el perfil en tres niveles:

* Bajo;
* Medio;
* Alto.

Además, muestra variables explicativas relacionadas con el comportamiento financiero del usuario.

### Gemelo digital

Permite simular escenarios financieros futuros modificando variables como:

* ingresos;
* gastos;
* capacidad de ahorro;
* horizonte temporal.

El objetivo es visualizar cómo podrían evolucionar las finanzas personales del usuario en distintos escenarios.

### Micro-ahorro

Calcula posibles estrategias de ahorro automático mediante redondeo de gastos.

Por ejemplo, si una compra es de 12,30 €, el sistema puede simular el ahorro de 0,70 € hasta redondear a 13,00 €.

---

## 2. Estructura del proyecto

```bash
PREFIN/
├── app.py                  # Frontend de la aplicación en Dash
├── main.py                 # Backend FastAPI para la conexión con TrueLayer
├── config.py               # Configuración visual y constantes
├── requirements.txt        # Dependencias necesarias
├── .env.example            # Ejemplo de variables de entorno para TrueLayer
├── assets/
│   └── custom.css          # Estilos CSS de la interfaz
├── fuentes/
│   ├── generator.py        # Generador de datos sintéticos
│   └── loader.py           # Carga de datos desde CSV, Excel o TrueLayer
└── modulos/
    ├── analyzer.py         # Cálculo de KPIs, tendencias y anomalías
    ├── categorizer.py      # Categorización automática de transacciones
    ├── digital_twin.py     # Simulación de escenarios financieros
    ├── microsavings.py     # Cálculo de micro-ahorro
    └── ml_model.py         # Modelo de Machine Learning para riesgo financiero
```

---

## 3. Instalación

### 3.1. Descargar el proyecto

Descomprimir la carpeta del proyecto y abrir una terminal dentro de la carpeta principal:

```bash
cd PREFIN
```

### 3.2. Crear un entorno virtual

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3. Instalar dependencias

Con el entorno virtual activado, ejecutar:

```bash
pip install -r requirements.txt
```

---

## 4. Ejecución de la aplicación

La aplicación tiene dos partes:

1. **Frontend Dash**, que muestra la interfaz web.
2. **Backend FastAPI**, necesario únicamente para la conexión con TrueLayer.

Para usar la aplicación con datos sintéticos o con CSV/Excel, basta con lanzar el frontend.

Para usar la conexión bancaria con TrueLayer, hay que lanzar frontend y backend en dos terminales distintas.

---

## 5. Opción recomendada para probar la aplicación: datos sintéticos

Esta es la forma más sencilla de ejecutar el proyecto, ya que no requiere credenciales de TrueLayer.

En una terminal, dentro de la carpeta `PREFIN`, ejecutar:

```bash
python app.py
```

Después, abrir en el navegador:

```text
http://localhost:8050
```

Una vez abierta la aplicación:

1. Ir a la pestaña **Datos**.
2. Pulsar **Generar datos sintéticos**.
3. Navegar por las secciones:

   * Dashboard;
   * Análisis;
   * Riesgo ML;
   * Gemelo Digital;
   * Micro-Ahorro.

Esta opción permite evaluar la funcionalidad completa de la aplicación sin necesidad de conectar una cuenta bancaria real.

---

## 6. Ejecución completa con TrueLayer Sandbox

Para probar la conexión bancaria mediante TrueLayer, es necesario ejecutar dos procesos al mismo tiempo: el backend y el frontend.

### 6.1. Configurar variables de entorno

Copiar el archivo `.env.example` y renombrarlo como `.env`.

En Windows:

```bash
copy .env.example .env
```

En macOS o Linux:

```bash
cp .env.example .env
```

Después, editar el archivo `.env` e introducir las credenciales de TrueLayer Sandbox:

```env
TRUELAYER_CLIENT_ID=tu_client_id
TRUELAYER_CLIENT_SECRET=tu_client_secret
TRUELAYER_REDIRECT_URI=http://localhost:8000/callback
```

El `redirect_uri` configurado en TrueLayer debe coincidir con:

```text
http://localhost:8000/callback
```

---

### 6.2. Abrir dos terminales

Para que la conexión con TrueLayer funcione correctamente, deben estar activos tanto el backend como el frontend.

### Terminal 1 — Backend FastAPI

En la primera terminal, dentro de la carpeta `PREFIN`, ejecutar:

```bash
uvicorn main:app --reload --port 8000
```

El backend quedará disponible en:

```text
http://localhost:8000
```

Esta parte gestiona la autenticación con TrueLayer y la obtención de transacciones bancarias.

---

### Terminal 2 — Frontend Dash

En una segunda terminal, también dentro de la carpeta `PREFIN`, activar el entorno virtual y ejecutar:

En Windows:

```bash
.venv\Scripts\activate
python app.py
```

En macOS o Linux:

```bash
source .venv/bin/activate
python app.py
```

El frontend quedará disponible en:

```text
http://localhost:8050
```

Esta es la dirección que debe abrirse en el navegador para utilizar la aplicación.

---

## 7. Resumen rápido de ejecución

### Modo demo, sin TrueLayer

Solo requiere una terminal:

```bash
python app.py
```

Abrir:

```text
http://localhost:8050
```

### Modo completo, con TrueLayer

Requiere dos terminales abiertas al mismo tiempo.

Terminal 1:

```bash
uvicorn main:app --reload --port 8000
```

Terminal 2:

```bash
python app.py
```

Abrir:

```text
http://localhost:8050
```

---

## 8. Carga de archivos CSV o Excel

La aplicación permite cargar un archivo de transacciones en formato CSV o Excel.

El archivo debe contener, como mínimo, las siguientes columnas:

| Columna     | Descripción                              | Ejemplo    |
| ----------- | ---------------------------------------- | ---------- |
| fecha       | Fecha de la transacción                  | 15/01/2026 |
| descripcion | Concepto o descripción de la transacción | Mercadona  |
| importe     | Importe de la transacción                | -32,50     |
| divisa      | Moneda de la transacción, opcional       | EUR        |

Los importes negativos representan gastos.

Los importes positivos representan ingresos.

Ejemplo:

```csv
fecha,descripcion,importe,divisa
01/01/2026,Nómina,1800,00,EUR
03/01/2026,Mercadona,-32,50,EUR
05/01/2026,Netflix,-12,99,EUR
```

Si el archivo procede de un banco, puede ser necesario adaptar los nombres de las columnas para que coincidan con los nombres esperados por la aplicación.

---

## 9. Tecnologías utilizadas

La aplicación se ha desarrollado con las siguientes tecnologías:

| Parte                | Tecnología                |
| -------------------- | ------------------------- |
| Frontend             | Dash                      |
| Componentes visuales | Dash Bootstrap Components |
| Gráficos             | Plotly                    |
| Backend              | FastAPI                   |
| Servidor backend     | Uvicorn                   |
| Machine Learning     | scikit-learn              |
| Tratamiento de datos | pandas, numpy             |
| Archivos Excel       | openpyxl                  |
| Open Banking         | TrueLayer Sandbox         |

---

## 10. Notas importantes

* Para evaluar el proyecto no es obligatorio utilizar TrueLayer.
* La aplicación puede probarse completamente con datos sintéticos.
* El frontend se ejecuta en el puerto `8050`.
* El backend se ejecuta en el puerto `8000`.
* Si se utiliza TrueLayer, ambos procesos deben estar activos a la vez.
* Si solo se utilizan datos sintéticos o CSV/Excel, no hace falta ejecutar el backend.
* El archivo `.env` solo es necesario para la conexión con TrueLayer.

---

## 11. Problemas frecuentes

### No se abre la aplicación

Comprobar que se ha ejecutado:

```bash
python app.py
```

y abrir en el navegador:

```text
http://localhost:8050
```

### El puerto 8050 está ocupado

Cerrar otros procesos que estén usando ese puerto o modificar el puerto en `app.py`.

### TrueLayer no conecta

Comprobar que:

1. el backend está ejecutándose en `http://localhost:8000`;
2. el frontend está ejecutándose en `http://localhost:8050`;
3. el archivo `.env` contiene las credenciales correctas;
4. el redirect URI configurado en TrueLayer es `http://localhost:8000/callback`.

### La aplicación funciona con datos sintéticos pero no con TrueLayer

Esto indica que el frontend funciona correctamente. El problema probablemente está en la configuración de TrueLayer, las credenciales o el backend.

---

## 12. Finalidad del proyecto

El objetivo de PREFIN es demostrar cómo una plataforma inteligente puede ayudar al usuario a comprender mejor su situación financiera, detectar patrones de comportamiento, anticipar riesgos y simular decisiones futuras.

La aplicación combina técnicas de análisis de datos, visualización, Machine Learning y simulación para ofrecer una herramienta de educación y planificación financiera personal.
