
# Guardian Deployment Packager
# ZIPs essential files for VPS transfer

$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$zipName = "Guardian_Deploy_$timestamp.zip"
$exclude = @("__pycache__", ".git", "venv", "logs", "data", "*.mp4", "*.zip")

Write-Host "📦 Packaging Guardian AI for Deployment..." -ForegroundColor Cyan

# Files/Folders to include
$includes = @(
    "src",
    "dashboard", 
    "config", 
    "models",
    "live_loop_v3.py",
    "guardian_watchdog.py",
    "run_guardian.bat",
    "requirements.txt",
    "README.md"
)

# Create a Temp Directory
$tempDir = "temp_deploy_$timestamp"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

# Copy Files
foreach ($item in $includes) {
    if (Test-Path $item) {
        Write-Host "   + Copying $item..."
        Copy-Item -Path $item -Destination $tempDir -Recurse -Force
    }
    else {
        Write-Host "   ⚠️ Warning: $item not found!" -ForegroundColor Yellow
    }
}

# Create ZIP
Write-Host "📚 Compressing to $zipName..."
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipName -Force

# Cleanup
Remove-Item -Path $tempDir -Recurse -Force

Write-Host "✅ Deployment Package Created: $zipName" -ForegroundColor Green
Write-Host "   (Ready to upload to VPS)"
