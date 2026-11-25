# push_gist.ps1 <file1> [file2 ...] — upload rl/ files as a secret gist, print raw URLs
param([Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)][string[]]$Files)

$in = "protocol=https`nhost=github.com`n`n"
$out = $in | git credential fill 2>$null
$tok = ($out | Select-String '^password=(.+)$').Matches.Groups[1].Value
if (-not $tok) { Write-Error "no github token"; exit 1 }
$hdr = @{ Authorization = "Bearer $tok"; Accept = 'application/vnd.github+json' }

$filesObj = @{}
foreach ($f in $Files) {
    $path = Join-Path $PSScriptRoot $f
    $filesObj[$f] = @{ content = [IO.File]::ReadAllText($path) }
}
$body = @{ description = 'lime rl transfer'; public = $false; files = $filesObj } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Uri 'https://api.github.com/gists' -Method Post -Headers $hdr -Body $body -TimeoutSec 30
Write-Host "gist id: $($r.id)"
foreach ($f in $Files) {
    Write-Host "$f $($r.files.$f.raw_url)"
}
