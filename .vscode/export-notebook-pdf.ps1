param([Parameter(Mandatory)][string]$Notebook)

$ErrorActionPreference = 'Stop'
$notebookPath = (Resolve-Path -LiteralPath $Notebook).Path
if ([IO.Path]::GetExtension($notebookPath) -ne '.ipynb') {
    throw 'Open and save a Jupyter notebook before running this task.'
}

$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$chromium = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe')
    (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe')
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$chromium) { throw 'Microsoft Edge or Google Chrome is required to print the notebook.' }

$tempDir = New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString()))
$profile = Join-Path $tempDir.FullName 'chromium-profile'
$pdf = [IO.Path]::ChangeExtension($notebookPath, '.pdf')
$rendered = Join-Path $tempDir.FullName 'notebook.pdf'
try {
    & $python -m nbconvert --to html $notebookPath --output notebook --output-dir $tempDir.FullName
    if ($LASTEXITCODE) { throw 'Jupyter HTML conversion failed.' }

    $htmlUri = [uri]::new((Join-Path $tempDir.FullName 'notebook.html')).AbsoluteUri
    $arguments = @('--headless', '--no-pdf-header-footer', '--no-first-run',
                   ('--user-data-dir="{0}"' -f $profile),
                   ('--print-to-pdf="{0}"' -f $rendered), $htmlUri)
    $process = Start-Process -FilePath $chromium -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0 -or !(Test-Path -LiteralPath $rendered)) {
        throw 'Chromium could not render the notebook to PDF.'
    }
    Copy-Item -LiteralPath $rendered -Destination $pdf -Force
    Write-Output "Saved $pdf"
}
finally {
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    $resolved = (Resolve-Path -LiteralPath $tempDir.FullName).Path
    if ($resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        [IO.Path]::GetFileName($resolved) -match '^[0-9a-f-]{36}$') {
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
    }
}

