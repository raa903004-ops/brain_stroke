$download = Join-Path $PSScriptRoot 'soop_download.log'
$training = Join-Path $PSScriptRoot 'soop_training.log'
$seen = @{}
Write-Host 'SOOP download and 150-epoch GPU training. Ctrl+C stops viewing only.' -ForegroundColor Cyan
while ($true) {
    foreach ($path in @($download, $training)) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $lines = @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)
        if (-not $seen.ContainsKey($path)) {
            $seen[$path] = [math]::Max(0, $lines.Count - 12)
        }
        if ($lines.Count -lt $seen[$path]) { $seen[$path] = 0 }
        for ($i = $seen[$path]; $i -lt $lines.Count; $i++) {
            Write-Host $lines[$i]
        }
        $seen[$path] = $lines.Count
    }
    Start-Sleep -Seconds 3
}
