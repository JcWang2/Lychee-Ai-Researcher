# Upload v2.5.0 (declarative method architecture: _TEMPLATE_REGISTRY,
# method_selector.py with ExperienceTable prior, planner PRIOR KNOWLEDGE
# block, evaluator/stage/portfolio lookup tables, test_v2_250.py) to A100
# and verify.
# NOTE: v2.5.0 is NOT pushed to the git repo (delivery-only big version).
# Run on the Windows workstation. Requires interactive SSH password for xzr.
$ErrorActionPreference = "Stop"

$SshHostName = "219.223.251.156"
$SshPort = 9000
$SshUser = "xzr"
$DeliveryDir = $PSScriptRoot
$PackageName = "ai_scientist_execution_layer_v2_20260807_v250.tar.gz"
$ExpectedSha256 = "552115BD96C2D02ED5EE9BB78E4EDA3C1D7DB20DD7638EC902635EE7E9BACF19"

$Package = Get-Item -LiteralPath (Join-Path $DeliveryDir $PackageName)
$ObservedSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Package.FullName).Hash.ToUpperInvariant()
if ($ObservedSha256 -ne $ExpectedSha256) {
    throw "Package SHA256 mismatch: observed=$ObservedSha256 expected=$ExpectedSha256"
}
$Sidecar = Join-Path $DeliveryDir "$($Package.Name).sha256"
if (-not (Test-Path -LiteralPath $Sidecar)) {
    throw "Package sidecar missing: $Sidecar"
}

$Remote = "${SshUser}@${SshHostName}"
ssh -p $SshPort $Remote "mkdir -p /mnt/data/stage42_delivery/incoming && mkdir -p /mnt/data/v2_hf_cache"
if ($LASTEXITCODE -ne 0) { throw "Remote incoming directory creation failed" }

Push-Location -LiteralPath $DeliveryDir
try {
    scp -P $SshPort $Package.Name (Split-Path -Leaf $Sidecar) `
        "run_v2_a100_3tasks_v23.sh" "run_v2_a100_lite_v250.sh" `
        "monitor_v2_v23.sh" "monitor_v2_v250_live.sh" `
        "${Remote}:/mnt/data/stage42_delivery/incoming/"
    if ($LASTEXITCODE -ne 0) { throw "Upload failed" }
}
finally {
    Pop-Location
}

ssh -p $SshPort $Remote "cd /mnt/data/stage42_delivery/incoming && echo '$ExpectedSha256  $PackageName' | sha256sum -c - && bash -n run_v2_a100_3tasks_v23.sh && bash -n run_v2_a100_lite_v250.sh && bash -n monitor_v2_v23.sh && bash -n monitor_v2_v250_live.sh"
if ($LASTEXITCODE -ne 0) { throw "Remote verification failed" }

Write-Host "UPLOAD_VERIFIED=YES"
Write-Host "PACKAGE_SHA256=$ObservedSha256"
Write-Host "Next (on the server) install v2.5.0 into the deploy tree:"
Write-Host "  cd /mnt/data/stage42_delivery/incoming"
Write-Host "  tar -xzf ai_scientist_execution_layer_v2_20260807_v250.tar.gz"
Write-Host "  cd ai_scientist_execution_layer_v2_20260807_v250 && sha256sum -c MANIFEST.sha256"
Write-Host "  export DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2"
Write-Host "  bash install_v2_execution_layer.sh --target `$DEPLOY_ROOT/MLE-bench/agents/aisci --run-tests"
Write-Host '  # expect V2_PACKAGE_MANIFEST=PASS / V2_PYCOMPILE=PASS / V2_OFFLINE_TESTS=PASS (incl. test_v2_250) / V2_INSTALL_VERIFY=PASS'
Write-Host "  # Then retest (TASK_LIST overridable, default = 3 image tasks):"
Write-Host "  #   nohup bash run_v2_a100_lite_v250.sh > run_v2_lite_v250_outer.log 2>&1 &"
Write-Host "  #   bash monitor_v2_v250_live.sh"
Write-Host "  # Acceptance: V2_INSTALL_VERIFY=PASS; profile:/grant:/receipt: verdict=success; NEW BEST."
ssh -p $SshPort $Remote "echo READY"
