@echo off
echo ============================================
echo   AmbulanceRoutingApp - Inicio del sistema
echo ============================================

REM Activar entorno virtual
echo Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias automáticamente
echo Instalando dependencias...
pip install -r requirements.txt

REM Ejecutar la aplicación
echo Iniciando servidor FastAPI...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
