# Run in PowerShell after extracting the project. Uses your own local Git login.
$ErrorActionPreference = "Stop"
$projectSource = $PSScriptRoot
$checkoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ("HW2_publish_" + [guid]::NewGuid().ToString("N"))
$repoUrl = "https://github.com/aghoniln/AIoT_L3_CWA_HW1.git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Please install Git for Windows first." }
git clone $repoUrl $checkoutPath
if ($LASTEXITCODE -ne 0) { throw "Git clone failed." }
Push-Location $checkoutPath
try {
    git switch -c hw2-multiple-linear-regression
    if ($LASTEXITCODE -ne 0) { throw "Could not create local branch." }
    $destination = Join-Path $checkoutPath "HW2"
    New-Item -ItemType Directory -Path $destination | Out-Null
    Get-ChildItem -LiteralPath $projectSource -Force | Where-Object {
        $_.Name -notin @(".git", ".venv", "__pycache__", ".pytest_cache")
    } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force }
    git add -- HW2
    git commit -m "Add HW2 multiple linear regression analysis and interactive web app"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed. Check git user.name and user.email." }
    git push -u origin hw2-multiple-linear-regression
    if ($LASTEXITCODE -ne 0) { throw "Push failed. Check your GitHub sign-in and repository write access. Checkout retained at $checkoutPath" }
    Write-Host "Uploaded: https://github.com/aghoniln/AIoT_L3_CWA_HW1/tree/hw2-multiple-linear-regression/HW2"
} finally { Pop-Location }
