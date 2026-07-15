# Publish AeroForesight-EU to GitHub.
# You log in once in your browser (GitHub device flow); this script does the rest.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\publish_to_github.ps1

$ErrorActionPreference = 'Stop'
$RepoName   = 'AeroForesight-EU'
$Visibility = '--public'          # change to --private if you prefer

# 1. Locate gh (portable install first, then PATH).
$ghPortable = Join-Path $env:LOCALAPPDATA 'aeroforesight-tools\bin\gh.exe'
$gh = if (Test-Path $ghPortable) { $ghPortable }
      elseif (Get-Command gh -ErrorAction SilentlyContinue) { 'gh' }
      else { Write-Error 'GitHub CLI (gh) not found.'; exit 1 }

# 2. Move to the repo root (parent of this script's folder).
Set-Location (Split-Path $PSScriptRoot -Parent)

# 3. Log in if needed (opens your browser; nothing is shared with anyone else).
& $gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n== Logging you in to GitHub (browser will open) ==" -ForegroundColor Cyan
    & $gh auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) { Write-Error 'Login failed or was cancelled.'; exit 1 }
}

# 4. Create the repo (if it exists already, just add the remote) and push.
$user = (& $gh api user --jq '.login').Trim()
Write-Host "`n== Publishing as $user ==" -ForegroundColor Cyan

if (-not (git remote 2>$null | Select-String -Quiet '^origin$')) {
    try {
        & $gh repo create "$RepoName" $Visibility --source=. --remote=origin --push
    } catch {
        # Repo may already exist on the account — wire it up and push directly.
        git remote add origin "https://github.com/$user/$RepoName.git"
        git push -u origin main
    }
} else {
    git push -u origin main
}

Write-Host "`n== Done ==  https://github.com/$user/$RepoName" -ForegroundColor Green
