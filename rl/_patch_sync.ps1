$p = 'C:\Users\c0a2504494\Documents\dojo\lime\lime_experiment\rl\train_sac_her.py'
$s = [System.IO.File]::ReadAllText($p)
# normalize to LF for reliable patching
$s = $s -replace "`r`n", "`n"

$sync = @'
    def _sync_artifacts(self):
        """Push artifacts to a secret gist so runtime death never loses progress. NEVER raises."""
        try:
            import json, urllib.request, base64 as b64mod
            gist_id = os.environ.get('ARTIFACT_GIST', '')
            tok_path = '/content/work/.ghtok'
            if not gist_id or not os.path.exists(tok_path):
                return
            tok = open(tok_path).read().strip()
            files = {}
            files['eval_history.json'] = {'content': json.dumps(self.history)}
            files['train_meta.json'] = {'content': json.dumps({'task': self.task, 'steps': self.num_timesteps, 'ts': time.time()})}
            cpng = os.path.join(self.out_dir, 'curve.png')
            if os.path.exists(cpng):
                files['curve_png.b64'] = {'content': b64mod.b64encode(open(cpng, 'rb').read()).decode()}
            ckdir = os.path.join(self.out_dir, 'checkpoints')
            if os.path.isdir(ckdir):
                zips = sorted(f for f in os.listdir(ckdir) if f.endswith('.zip'))
                if zips and zips[-1] != getattr(self, '_synced_ckpt', None):
                    raw = open(os.path.join(ckdir, zips[-1]), 'rb').read()
                    if len(raw) < 5000000:
                        key = 'ckpt_' + zips[-1].replace('.zip', '') + '.b64'
                        files[key] = {'content': b64mod.b64encode(raw).decode()}
                        files['ckpt_latest.txt'] = {'content': zips[-1]}
                        self._synced_ckpt = zips[-1]
            body = json.dumps({'files': files}).encode()
            req = urllib.request.Request('https://api.github.com/gists/' + gist_id, data=body, method='PATCH',
                headers={'Authorization': 'Bearer ' + tok, 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=30).read()
            print('[sync] artifacts pushed at t=%d' % self.num_timesteps)
        except Exception as e:
            print('[sync] failed (non-fatal):', e)

'@

$anchor = "    def _save_curve(self):"
if (-not $s.Contains($anchor)) { Write-Host 'ANCHOR-MISSING'; exit 1 }
$s = $s.Replace($anchor, $sync + $anchor)

$callsite = "        self._save_curve()`n"
if (-not $s.Contains($callsite)) { Write-Host 'CALLSITE-MISSING'; exit 1 }
$s = $s.Replace($callsite, "        self._sync_artifacts()`n" + $callsite)

if (-not $s.Contains("`nimport time`n")) {
    $s = $s.Replace("import argparse`n", "import argparse`nimport time`n")
}

$tail = 'model.save(os.path.join(args.out, "final_model"))'
if (-not $s.Contains($tail)) { Write-Host 'TAIL-MISSING'; exit 1 }
$s = $s.Replace($tail, $tail + "`n    cb._sync_artifacts()")

# here-string indentation fix: the sync block got zero-padded correctly already since col 0 in file
[System.IO.File]::WriteAllText($p, $s, (New-Object System.Text.UTF8Encoding $false))
Write-Host 'PATCH-WRITTEN'
Write-Host ('sync present: ' + $s.Contains('_sync_artifacts'))
Write-Host ('finalsync present: ' + $s.Contains('cb._sync_artifacts()'))
