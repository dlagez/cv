param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [ValidateSet("predict", "track")]
    [string]$Mode = "predict",

    [string]$Model = "yolo11n.pt",

    [double]$Conf = 0.30,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$yolo = Join-Path $PSScriptRoot ".venv\Scripts\yolo.exe"

if (-not (Test-Path $yolo)) {
    throw "YOLO CLI not found. Activate or recreate the virtual environment in $PSScriptRoot\.venv first."
}

$args = @(
    $Mode,
    "model=$Model",
    "source=$Source",
    "classes=0",
    "conf=$Conf",
    "save=True"
)

if ($ExtraArgs) {
    $args += $ExtraArgs
}

& $yolo @args
exit $LASTEXITCODE
