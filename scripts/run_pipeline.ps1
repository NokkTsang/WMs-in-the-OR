# Feature extraction (all encoders x cameras), then the analysis notebooks (01 probes, 02 dynamics, 03 synthesis) per procedure.
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 [-Procedure MISS] [-Notebooks 02_dynamics_and_anomaly] [-SkipNotebook] [-SkipExtract]
# Existing feature files are skipped, so the script is resumable. Encoders whose checkpoint is missing in ckpts\ are skipped.
# Two processes (one per procedure) can run in parallel.
param(
    [ValidateSet('MISS', 'Ultrasound')][string[]]$Procedure = @('MISS', 'Ultrasound'),
    [string[]]$Notebooks = @('01_probe_and_temporal', '02_dynamics_and_anomaly', '03_synthesis', '04_predictor_comparison'),
    [switch]$SkipNotebook,
    [switch]$SkipExtract
)
$ErrorActionPreference = 'Stop'
$Notebooks = @($Notebooks | ForEach-Object { $_ -split ',' } | Where-Object { $_ })  # `-File` passes "a,b" as one string
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'

function Assert-Ok($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

# Ultrasound external_* exist only in ultrasound_3/4/5_* (18 of 25 takes); extraction skips takes without the camera.
$cams = [ordered]@{
    MISS       = @('external_3', 'external_5', 'external_1', 'external_2', 'external_4', 'microscope')
    Ultrasound = @('simstation', 'or_light', 'external_3')
}
$encoders = @(
    @{ name = 'dinov2';        ckpt = $null },
    @{ name = 'caformer_in1k'; ckpt = 'ckpts\caformer_s18_in1k.pth' },
    @{ name = 'surgenet';      ckpt = 'ckpts\surgenetxl_teacher.pth' },
    @{ name = 'vjepa2';        ckpt = 'ckpts\vjepa2_vitl.pt' },
    @{ name = 'surgmotion';    ckpt = 'ckpts\SurgMotion-vitl.pt' }
)
if (-not $SkipExtract) {
    $available = @($encoders | Where-Object { -not $_.ckpt -or (Test-Path $_.ckpt) })
    foreach ($e in $encoders) { if ($e -notin $available) { Write-Host "SKIP $($e.name): $($e.ckpt) missing" } }
    $names = @($available | ForEach-Object { $_.name })
    foreach ($proc in $Procedure) {
        foreach ($cam in $cams[$proc]) {
            # one pass per camera: the frames are read once and shared by all encoders (reading dominates the run time)
            & $py scripts\extract_features.py --model $names --cam $cam --procedure $proc
            Assert-Ok "extraction $cam"
        }
    }
}

if (-not $SkipNotebook) {
    Push-Location notebooks
    foreach ($proc in $Procedure) {
        $env:EGOEXOR_PROC = $proc
        foreach ($nb in $Notebooks) {
            $out = "${nb}_$proc.ipynb"  # one file per procedure, so parallel runs never share a notebook
            & (Join-Path $root '.venv\Scripts\jupytext.exe') --to ipynb --output $out "$nb.py" | Out-Null
            & (Join-Path $root '.venv\Scripts\jupyter.exe') nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=wm-or `
                --ExecutePreprocessor.timeout=14400 $out
            Assert-Ok "$nb ($proc)"
        }
    }
    Pop-Location
}
Write-Host 'DONE: see outputs\probe, outputs\dynamics, outputs\synthesis and the executed notebooks in notebooks\'
