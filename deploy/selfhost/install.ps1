<#
.SYNOPSIS
  Install Legion Hub on Windows with Docker Desktop.

.DESCRIPTION
  Checks Docker, writes .env with fresh secrets, copies config\legion.toml,
  pulls the images and starts the stack. Safe to run again: an existing .env
  and config\legion.toml are never overwritten.

.PARAMETER Domain
  Serve the Hub on https://<Domain> with an automatic certificate (ports 80 and
  443 must reach this machine and DNS must point here).

.PARAMETER Version
  The release to run (default: latest).

.PARAMETER Build
  Build the images from this checkout instead of pulling them.

.PARAMETER NoStart
  Write the configuration and stop.

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Domain hub.example.com -Version v0.1.0
#>
[CmdletBinding()]
param(
  [string]$Domain = '',
  [string]$Version = '',
  [switch]$Build,
  [switch]$NoStart
)

$ErrorActionPreference = 'Stop'
$RepoUrl = if ($env:LEGION_REPO_URL) { $env:LEGION_REPO_URL } else { 'https://github.com/MyLegionOnline/legion-hub' }

function Write-Step([string]$Text) { Write-Host "`n==> $Text" }
function Fail([string]$Text) { Write-Error $Text; exit 1 }

# Run a native command and fail with the given message when it does not exit 0.
function Test-Native([string]$Message, [scriptblock]$Command) {
  try { & $Command *> $null } catch { Fail $Message }
  if ($LASTEXITCODE -ne 0) { Fail $Message }
}

# ── Where the files are ────────────────────────────────────────────────────
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -or -not (Test-Path (Join-Path $ScriptDir 'docker-compose.yml'))) {
  $Target = if ($env:LEGION_DIR) { $env:LEGION_DIR } else { Join-Path $HOME 'legion' }
  Write-Step "Downloading the install bundle to $Target"
  New-Item -ItemType Directory -Force -Path $Target | Out-Null
  $BundleVersion = if ($Version) { $Version } else { 'latest' }
  $Url = if ($BundleVersion -eq 'latest') { "$RepoUrl/releases/latest/download/legion-selfhost.zip" } else { "$RepoUrl/releases/download/$BundleVersion/legion-selfhost.zip" }
  $Zip = Join-Path ([System.IO.Path]::GetTempPath()) 'legion-selfhost.zip'
  Invoke-WebRequest -Uri $Url -OutFile $Zip
  Expand-Archive -Path $Zip -DestinationPath $Target -Force
  Remove-Item $Zip
  $ScriptDir = $Target
}
Set-Location $ScriptDir

# ── 1. Docker ──────────────────────────────────────────────────────────────
Write-Step 'Checking Docker'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Fail 'Docker is not installed. Install Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/'
}
Test-Native 'Docker is installed but not running. Start Docker Desktop and run this script again.' { docker info }
Test-Native "Docker Compose v2 is missing ('docker compose'). Update Docker Desktop." { docker compose version }
$EngineVersion = (docker version --format '{{.Server.Version}}').Trim()
if ([int]($EngineVersion.Split('.')[0]) -lt 26) {
  Fail "Docker Engine 26 or newer is required (found $EngineVersion): room containers mount their folder of a shared volume by sub-path. Update Docker Desktop."
}
Write-Host "Docker $EngineVersion and Compose are ready."

# ── 2. .env with generated secrets ─────────────────────────────────────────
function New-Secret {
  $bytes = New-Object byte[] 32
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  -join ($bytes | ForEach-Object { $_.ToString('x2') })
}

# .env is read by Docker Compose: UTF-8 without BOM, LF line ends.
function Write-EnvFile([string[]]$Lines) {
  $text = ($Lines -join "`n") + "`n"
  [System.IO.File]::WriteAllText((Join-Path $PWD '.env'), $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Set-EnvVar([string]$Name, [string]$Value) {
  $lines = @(Get-Content .env | Where-Object { $_ -notmatch "^$([regex]::Escape($Name))=" })
  Write-EnvFile ($lines + "$Name=$Value")
}

Write-Step 'Checking .env'
if (Test-Path .env) {
  Write-Host '.env already exists, leaving the secrets in it alone'
} else {
  Copy-Item .env.example .env
  foreach ($name in 'POSTGRES_PASSWORD', 'SECRET_KEY', 'WORKSPACED_TOKEN', 'MCP_HOST_TOKEN', 'MCP_GATEWAY_TOKEN') {
    Set-EnvVar $name (New-Secret)
  }
  Write-Host 'Created .env with new secrets (keep a copy somewhere safe).'
}
if ($Version) { Set-EnvVar 'LEGION_VERSION' $Version }
if ($Domain) {
  Set-EnvVar 'HUB_DOMAIN' $Domain
  Set-EnvVar 'HUB_PUBLIC_URL' "https://$Domain"
  Set-EnvVar 'COMPOSE_PROFILES' 'caddy'
  Write-Host "Configured https://$Domain"
}

# ── 3. config\legion.toml ──────────────────────────────────────────────────
Write-Step 'Checking config\legion.toml'
New-Item -ItemType Directory -Force -Path config | Out-Null
if (Test-Path config\legion.toml) {
  Write-Host 'config\legion.toml already exists, leaving it alone'
} else {
  $Example = @('legion.example.toml', '..\..\hub\legion.example.toml') | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $Example) { Fail 'legion.example.toml not found next to the installer' }
  Copy-Item $Example config\legion.toml
  Write-Host 'Created config\legion.toml (every key is commented out: the defaults apply).'
}

if ($NoStart) {
  Write-Step 'Configuration written. Start the Hub with:  docker compose up -d'
  exit 0
}

# ── 4. Images and start ────────────────────────────────────────────────────
if ($Build) {
  Write-Step 'Building the images from this checkout (this takes a while)'
  docker compose build
  if ($LASTEXITCODE -ne 0) { Fail 'the build failed' }
} else {
  Write-Step 'Pulling the images'
  docker compose pull
  if ($LASTEXITCODE -ne 0) { Fail 'could not pull the images. Check the connection; for an unreleased checkout run again with -Build.' }
}

Write-Step 'Starting Legion Hub'
docker compose up -d
if ($LASTEXITCODE -ne 0) { Fail 'docker compose up failed; see the output above' }

$PublicUrl = (Get-Content .env | Where-Object { $_ -match '^HUB_PUBLIC_URL=' } | Select-Object -First 1) -replace '^HUB_PUBLIC_URL=', ''
if (-not $PublicUrl) { $PublicUrl = 'http://localhost:8080' }
Write-Step 'Done'
Write-Host "  Open $PublicUrl"
Write-Host '  The first account you register becomes the owner of this Hub.'
Write-Host ''
Write-Host 'Manage it from this folder:'
Write-Host '  docker compose ps | logs -f backend | down | pull; docker compose up -d   (update)'
