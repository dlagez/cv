param(
    [Parameter(Mandatory = $true)]
    [string[]]$Source,

    [string]$Model = "yolo11n-pose.pt",
    [string]$OutputDir = "",
    [int]$Imgsz = 960,
    [double]$Conf = 0.25,
    [double]$KeypointConf = 0.25,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found: $python"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $scriptDir "outputs\pose-test"
}

$arguments = @(
    "-m", "helmet_classifier.pose_test",
    "--model", $Model,
    "--output-dir", $OutputDir,
    "--imgsz", $Imgsz,
    "--conf", $Conf,
    "--keypoint-conf", $KeypointConf
)

if (-not [string]::IsNullOrWhiteSpace($Device)) {
    $arguments += @("--device", $Device)
}

foreach ($item in $Source) {
    $arguments += @("--source", $item)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

