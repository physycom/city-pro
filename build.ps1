#!/usr/bin/env pwsh

#Remove-Item build -Force -Recurse -ErrorAction SilentlyContinue
New-Item -Path .\build_windows -ItemType directory -Force
Set-Location build_windows

cmake -G "Visual Studio 15 2017 Win64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET" ..
cmake --build . --config Release

Set-Location ..
