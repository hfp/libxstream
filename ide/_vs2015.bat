@ECHO OFF
SETLOCAL

SET LIBXSTREAM_ROOT=%~d0%~p0\..

CALL %~d0"%~p0"_vs.bat vs2015
START %~d0"%~p0"_vs2015.sln

ENDLOCAL