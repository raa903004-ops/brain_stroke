$prepLog = Join-Path $PSScriptRoot 'flair_preparation.log'
$trainingLog = Join-Path $PSScriptRoot 'training.log'
$extendedLog = Join-Path $PSScriptRoot 'training_100.log'
$resumeLog = Join-Path $PSScriptRoot 'training_resume.log'
$seen = @{}
Write-Host 'GPU training status. Ctrl+C stops viewing only.' -ForegroundColor Cyan
while ($true) {
    foreach ($path in @($resumeLog)) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $lines = @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)
        if (-not $seen.ContainsKey($path)) {
            $seen[$path] = 0
        }
        if ($lines.Count -lt $seen[$path]) { $seen[$path] = 0 }
        for ($i = $seen[$path]; $i -lt $lines.Count; $i++) {
            Write-Host $lines[$i]
        }
        $seen[$path] = $lines.Count
    }
    Start-Sleep -Seconds 2
}
