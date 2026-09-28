$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = 'D:\brain-stroke-gpu-venv\Scripts\python.exe'
$wheel = 'D:\brain-stroke-install\torch-2.6.0+cu118-cp312-cp312-win_amd64.whl'
$url = 'https://download-r2.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp312-cp312-win_amd64.whl'
$sha256 = '6c040e4181c5dae73b965b61394ec431c93b2018165e2be8f15fc68d44444cb3'
$env:TEMP = 'D:\brain-stroke-install'
$env:TMP = 'D:\brain-stroke-install'
$env:PIP_CACHE_DIR = 'D:\brain-stroke-install\cache'

Write-Host 'Downloading official PyTorch 2.6 CUDA 11.8 wheel...' -ForegroundColor Cyan
curl.exe -L -C - --retry 20 --retry-all-errors --retry-delay 5 -o $wheel $url
if ($LASTEXITCODE -ne 0) { throw "CUDA wheel download failed: $LASTEXITCODE" }
if ((Get-Item -LiteralPath $wheel).Length -ne 2728858831) { throw 'CUDA wheel size mismatch' }
if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha256) { throw 'CUDA wheel checksum mismatch' }
Write-Host 'CUDA wheel verified. Installing...' -ForegroundColor Cyan
& $python -m pip install --force-reinstall $wheel 2>&1 | Tee-Object -FilePath 'cuda_install.log'
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed: $LASTEXITCODE" }
& $python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'device', torch.cuda.get_device_name(0)); x=torch.ones(1,device='cuda'); print('CUDA test',x.item())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed' }

Write-Host 'Starting five GPU training epochs...' -ForegroundColor Cyan
& $python -u train_local.py --epochs 5 2>&1 | Tee-Object -FilePath 'training.log'
if ($LASTEXITCODE -ne 0) { throw "GPU training failed: $LASTEXITCODE" }
Write-Host 'GPU training complete. Weights: weights\best_unet3d.pth' -ForegroundColor Green
