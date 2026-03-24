param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$Model = "D:\code-ai\cv\helmet_classifier-v2\models\best.pt",

    [string]$Output = "",

    [int]$Imgsz = 960,

    [double]$Conf = 0.25,

    [double]$PersonConf = 0.45,

    [double]$HardhatConf = 0.25,

    [string]$Device = "",

    [int]$LineWidth = 1,

    [double]$FontSize = 20,

    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "helmet_classifier-v2\ppe_detect_video.py"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found: $python"
}

if (-not (Test-Path $script)) {
    throw "Script not found: $script"
}

$arguments = @(
    $script,
    "--source", $Source,
    "--model", $Model,
    "--imgsz", $Imgsz,
    "--conf", $Conf,
    "--person-conf", $PersonConf,
    "--hardhat-conf", $HardhatConf,
    "--line-width", $LineWidth,
    "--font-size", $FontSize
)

if ($Output) {
    $arguments += @("--output", $Output)
}

if ($Device) {
    $arguments += @("--device", $Device)
}

if ($MaxFrames -gt 0) {
    $arguments += @("--max-frames", $MaxFrames)
}

& $python @arguments
exit $LASTEXITCODE
