$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = 'D:\brain-stroke-venv\Scripts\python.exe'

Write-Host 'ISLES 2022: загрузка, проверка и распаковка' -ForegroundColor Cyan
& $python -u prepare_isles.py 2>&1 | Tee-Object -FilePath 'download_isles.log'
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка подготовки датасета (код $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host 'Запуск обучения 3D U-Net на 5 эпох' -ForegroundColor Cyan
& $python -u train_local.py --epochs 5 2>&1 | Tee-Object -FilePath 'training.log'
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка обучения (код $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host 'Обучение завершено. Веса: weights\best_unet3d.pth' -ForegroundColor Green
