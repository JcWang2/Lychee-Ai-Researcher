#!/usr/bin/env bash
set -euo pipefail

readonly PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly PAYLOAD_ROOT="$PACKAGE_ROOT/payload/agents/aisci"
readonly FILE_LIST="$PACKAGE_ROOT/PAYLOAD_FILES.txt"
readonly MANIFEST="$PACKAGE_ROOT/MANIFEST.sha256"

target_root=""
install_target=1
verify_only=0
run_tests=0
python_bin=""

usage() {
    cat <<'USAGE'
Usage:
  install_v2_execution_layer.sh [options]

Options:
  --target PATH       Agent source root. Default: $DEPLOY_ROOT/MLE-bench/agents/aisci
  --no-target         Do not install into the deployment tree.
  --verify-only       Verify manifest + compare installed tree with this payload; do not copy.
  --run-tests         Run the offline V2 test suite after target installation.
                      (heavy compiled-harness e2e skipped by default; set
                      V2_TEST_HARNESS_SKIP=0 to include it; default timeout
                      V2_TEST_HARNESS_TIMEOUT=1800)
  --python PATH       Python executable for py_compile/import/test checks.
  -h, --help          Show this help.

The package must be extracted before this script is run. MANIFEST.sha256 is
verified before any file is copied. Only files enumerated in PAYLOAD_FILES.txt
are overlaid; existing target files are backed up under
<target>/.v2_backup_<timestamp>.
USAGE
}

fail() {
    printf 'V2_INSTALL=FAIL:%s\n' "$*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)
            [ "$#" -ge 2 ] || fail "missing_value_for_target"
            target_root="$2"
            shift 2
            ;;
        --no-target)
            install_target=0
            shift
            ;;
        --verify-only)
            verify_only=1
            shift
            ;;
        --run-tests)
            run_tests=1
            shift
            ;;
        --python)
            [ "$#" -ge 2 ] || fail "missing_value_for_python"
            python_bin="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown_argument:$1"
            ;;
    esac
done

[ -f "$FILE_LIST" ] || fail "payload_file_list_missing"
[ -f "$MANIFEST" ] || fail "manifest_missing"
[ -d "$PAYLOAD_ROOT" ] || fail "payload_root_missing"

if [ -z "$python_bin" ]; then
    python_bin="$(command -v python3 || command -v python || true)"
fi
[ -n "$python_bin" ] || fail "no_python_found"

if [ -z "$target_root" ] && [ "$install_target" -eq 1 ]; then
    [ -n "${DEPLOY_ROOT:-}" ] || fail "DEPLOY_ROOT_not_set_and_no_target_given"
    target_root="$DEPLOY_ROOT/MLE-bench/agents/aisci"
fi
[ "$verify_only" -eq 1 ] || [ "$install_target" -eq 1 ] || fail "no_install_destination_selected"
[ "$run_tests" -eq 0 ] || [ "$install_target" -eq 1 ] || fail "run_tests_requires_target"

verify_package() {
    (
        cd "$PACKAGE_ROOT"
        sha256sum -c "$(basename "$MANIFEST")"
    ) || fail "package_manifest_mismatch"
    while IFS= read -r relative_path; do
        [ -n "$relative_path" ] || continue
        case "$relative_path" in
            /*|*".."*) fail "unsafe_payload_path:$relative_path" ;;
        esac
        [ -f "$PAYLOAD_ROOT/$relative_path" ] || fail "payload_file_missing:$relative_path"
    done < "$FILE_LIST"
    printf 'V2_PACKAGE_MANIFEST=PASS\n'
}

compare_tree() {
    local dest="$1"
    local mismatches=0
    while IFS= read -r relative_path; do
        [ -n "$relative_path" ] || continue
        if ! cmp -s "$PAYLOAD_ROOT/$relative_path" "$dest/$relative_path"; then
            printf 'V2_MISMATCH:%s\n' "$relative_path" >&2
            mismatches=$((mismatches + 1))
        fi
    done < "$FILE_LIST"
    [ "$mismatches" -eq 0 ] || fail "tree_mismatch:$mismatches"
    printf 'V2_TREE_VERIFY=PASS\n'
}

install_payload() {
    local dest="$1"
    [ -d "$dest" ] || fail "destination_missing:$dest"
    local stamp
    stamp="$(date +%Y%m%dT%H%M%SZ)"
    local backup_root="$dest/.v2_backup_$stamp"
    while IFS= read -r relative_path; do
        [ -n "$relative_path" ] || continue
        local src="$PAYLOAD_ROOT/$relative_path"
        local dst_file="$dest/$relative_path"
        if [ -f "$dst_file" ]; then
            mkdir -p "$backup_root/$(dirname "$relative_path")"
            cp -p "$dst_file" "$backup_root/$relative_path"
        fi
        mkdir -p "$(dirname "$dst_file")"
        cp -p "$src" "$dst_file"
    done < "$FILE_LIST"
    printf 'V2_BACKUP_ROOT=%s\n' "$backup_root"
    printf 'V2_INSTALL=OK\n'
}

compile_payload() {
    local dest="$1"
    while IFS= read -r relative_path; do
        [ -n "$relative_path" ] || continue
        case "$relative_path" in
            *.py)
                "$python_bin" -m py_compile "$dest/$relative_path" \
                    || fail "py_compile_failed:$relative_path"
                ;;
        esac
    done < "$FILE_LIST"
    printf 'V2_PYCOMPILE=PASS\n'
}

run_offline_tests() {
    local dest="$1"
    local missing=0
    local test
    local TESTS="test_v2_metrics.py test_v2_contracts.py test_v2_pact.py test_v2_hera.py \
                 test_v2_stage_controller.py test_v2_resource_profiler.py \
                 test_v2_l1_transactional.py test_v2_closed_loop.py test_v2_23.py \
                 test_v2_234.py test_v2_235.py test_v2_236.py test_v2_237.py \
                 test_v2_240.py test_v2_238.py test_v2_239.py test_v2_250.py \
                 test_v2_251.py test_v2_252.py test_v2_254.py test_v2_255.py"
    # Guard: every test in the run list must be shipped in the payload AND
    # actually installed into the target. Catches packaging omissions early
    # (e.g. a new test file missing from PAYLOAD_FILES.txt) with a clear
    # message instead of a confusing "No such file or directory" exit.
    for test in $TESTS; do
        if ! grep -qx "$test" "$FILE_LIST"; then
            printf 'V2_INSTALL=FAIL:test_missing_from_payload:%s\n' "$test" >&2
            missing=$((missing + 1))
        fi
        if [ ! -f "$dest/$test" ]; then
            printf 'V2_INSTALL=FAIL:test_missing_in_target:%s\n' "$test" >&2
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] || exit 1
    for test in $TESTS; do
        (
            cd "$dest"
            "$python_bin" "$test"
        ) || fail "v2_offline_test_failed:$test"
    done
    printf 'V2_OFFLINE_TESTS=PASS\n'
}

verify_package

if [ "$verify_only" -eq 1 ]; then
    [ -n "$target_root" ] || fail "verify_only_requires_target"
    compare_tree "$target_root"
    exit 0
fi

install_payload "$target_root"
compile_payload "$target_root"
compare_tree "$target_root"
if [ "$run_tests" -eq 1 ]; then
    # Align with run_v2_a100_lite_v255.sh: on ops installs the heavy
    # compiled-harness end-to-end subprocess (test_v2_251/test_v2_255) is
    # skipped by default; set V2_TEST_HARNESS_SKIP=0 to re-validate it, and
    # V2_TEST_HARNESS_TIMEOUT to raise/lower the subprocess timeout.
    export V2_TEST_HARNESS_TIMEOUT="${V2_TEST_HARNESS_TIMEOUT:-1800}"
    export V2_TEST_HARNESS_SKIP="${V2_TEST_HARNESS_SKIP:-1}"
    run_offline_tests "$target_root"
fi
printf 'V2_INSTALL_VERIFY=PASS\n'
