$download = Join-Path $PSScriptRoot 'download_isles.log'
$training = Join-Path $PSScriptRoot 'training.log'
$seen = @{}
Write-Host 'ISLES 2022 download and training log. Ctrl+C stops viewing only.' -ForegroundColor Cyan
while ($true) {
    foreach ($path in @($download, $training)) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $lines = @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)
        if (-not $seen.ContainsKey($path)) {
            $seen[$path] = [Math]::Max(0, $lines.Count - 8)
        }
        for ($i = $seen[$path]; $i -lt $lines.Count; $i++) {
            Write-Host $lines[$i]
        }
        $seen[$path] = $lines.Count
    }
    Start-Sleep -Seconds 2
}
