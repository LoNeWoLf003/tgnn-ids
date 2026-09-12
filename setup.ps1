$PROJECT_ROOT = "$HOME\tgnn-ids"
New-Item -ItemType Directory -Force -Path $PROJECT_ROOT
Set-Location $PROJECT_ROOT

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip