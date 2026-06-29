# API Documentation

Este proyecto es una API construida con FastAPI para la gestión de modelos de machine learning. La API permite realizar operaciones de entrenamiento, predicción y evaluación de modelos.

## Estructura del Proyecto

El proyecto tiene la siguiente estructura de directorios:

```
api/
├── app/
│   ├── main.py                # Punto de entrada de la aplicación FastAPI
│   ├── routers/               # Contiene los routers de la API
│   │   └── __init__.py        # Inicializa el paquete de routers
│   └── models/                # Contiene los modelos de datos
│       └── __init__.py        # Inicializa el paquete de modelos
├── requirements.txt            # Lista de dependencias del proyecto
└── README.md                   # Documentación del proyecto
```

## Instalación

Para instalar las dependencias del proyecto, ejecuta el siguiente comando:

```
pip install -r requirements.txt
```

## Ejecución

Para ejecutar la API, utiliza el siguiente comando:

```
uvicorn app.main:app --reload
```

Esto iniciará el servidor en modo de desarrollo, permitiendo recargas automáticas.

## Endpoints

La API incluye los siguientes endpoints:

- **Entrenamiento**: Rutas para iniciar el entrenamiento de modelos.
- **Predicción**: Rutas para realizar inferencias con modelos entrenados.
- **Evaluación**: Rutas para evaluar el rendimiento de los modelos.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o un pull request en el repositorio.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.