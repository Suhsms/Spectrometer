$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
        & $python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Could not install application dependencies.' }
    }
    & $python app.py
    if ($LASTEXITCODE -ne 0) { throw 'Simple Spectrometer exited with an error.' }
}
finally {
    Pop-Location
}