@echo off
rem Build sycl_engine.dll (clang-cl -fsycl) against the oneMath generic BLAS backend from the spike.
rem Expects the MSVC + open DPC++ environment (vcvars64 and DPC++ bin/lib on PATH/LIB);
rem ONEMATH_SRC / ONEMATH_BUILD point at the oneMath source and build trees.
if exist "%~dp0..\spike\env.cmd" call "%~dp0..\spike\env.cmd"
set "OM=%ONEMATH_SRC%"
if "%OM%"=="" set "OM=%~dp0..\oneMath"
set "OMB=%ONEMATH_BUILD%"
if "%OMB%"=="" set "OMB=%~dp0..\spike\build-win"
set "OUT=%~dp0build"
if not exist "%OUT%" mkdir "%OUT%"
clang-cl -fsycl /nologo /EHsc /O2 /std:c++17 -w -I"%OM%\include" -I"%OMB%\bin" ^
  "%~dp0engine\sycl_engine.cpp" "%OMB%\lib\onemath_blas_generic.lib" ^
  /link /dll /out:"%OUT%\sycl_engine.dll" /implib:"%OUT%\sycl_engine.lib" || exit /b 1
copy /y "%OMB%\bin\onemath_blas_generic.dll" "%OUT%\" >nul
echo ENGINE_BUILD_OK
