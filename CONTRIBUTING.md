# Contributing

Thanks for helping with the AI Scientist execution layer (v2.3.3).

## Repo layout

```
payload/agents/aisci/   runtime code (analyzer, compiler, registry, PACT, tests)
docs/                   architecture notes
install_v2_execution_layer.sh   deployment/verification script
MANIFEST.sha256         integrity manifest (do NOT edit by hand; regenerate when payload changes)
```

## Ground rules

- **No competition-name hardcoding.** All modality / task-type / capability decisions
  must be data-shape driven (content sampling, metrics registry, resource profile).
- Line endings must be LF for `.py` / `.sh`.
- `program_compiler.py` is generated from `parts/*.txt` templates by
  `make_compiler.py`; edit the parts, then regenerate, never hand-patch the
  generated file.
- When you change any file under `payload/`, regenerate `MANIFEST.sha256` and the
  deployment tarball before releasing.

## Testing

Run all 9 offline suites from `payload/agents/aisci` (each prints `RESULT=PASS`):

```bash
for t in test_v2_metrics.py test_v2_contracts.py test_v2_pact.py test_v2_hera.py \
         test_v2_stage_controller.py test_v2_resource_profiler.py \
         test_v2_l1_transactional.py test_v2_closed_loop.py test_v2_23.py; do
  python "$t" || exit 1
done
```

## Deployment verification

```bash
sha256sum -c MANIFEST.sha256
bash install_v2_execution_layer.sh --target <deploy>/MLE-bench/agents/aisci --run-tests
# expect V2_PACKAGE_MANIFEST=PASS / V2_PYCOMPILE=PASS / V2_OFFLINE_TESTS=PASS / V2_INSTALL_VERIFY=PASS
```