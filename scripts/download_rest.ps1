# Resumable download of the remaining Ultrasound files (~117 GB). Safe to re-run; re-running continues where it stopped.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& .venv\Scripts\python.exe scripts\download_segmented.py ultrasound_3.h5 ultrasound_4.h5 ultrasound_5_14.h5 ultrasound_5_58.h5 --segments 8 *>&1 | Tee-Object -FilePath data\download.log
