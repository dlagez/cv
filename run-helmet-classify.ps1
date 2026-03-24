param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$Model = "yolo11n-pose.pt",

    [string]$Output = "",

    [double]$PersonConf = 0.35,

    [double]$WhiteRatioThreshold = 0.18,

    [double]$RedRatioThreshold = 0.06,

    [int]$Imgsz = 960,

    [string]$FontPath = "",

    [int]$FontSize = 22,

    [switch]$DrawHelmetBox,

    [switch]$DebugText,

    [int]$MaxFrames = 0,

    [string]$Device = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "helmet_color_classify.py"

if (-not (Test-Path $python)) {
    throw "Python executable not found in $PSScriptRoot\.venv"
}

if (-not (Test-Path $script)) {
    throw "Script not found: $script"
}

$args = @(
    $script,
    "--source",
    $Source,
    "--model",
    $Model,
    "--person-conf",
    $PersonConf,
    "--white-ratio-threshold",
    $WhiteRatioThreshold,
    "--red-ratio-threshold",
    $RedRatioThreshold,
    "--imgsz",
    $Imgsz,
    "--font-size",
    $FontSize
)

if ($Output) {
    $args += @("--output", $Output)
}

if ($FontPath) {
    $args += @("--font-path", $FontPath)
}

if ($DrawHelmetBox) {
    $args += "--draw-helmet-box"
}

if ($DebugText) {
    $args += "--debug-text"
}

if ($MaxFrames -gt 0) {
    $args += @("--max-frames", $MaxFrames)
}

if ($Device) {
    $args += @("--device", $Device)
}

if ($ExtraArgs) {
    $args += $ExtraArgs
}

& $python @args
exit $LASTEXITCODE
