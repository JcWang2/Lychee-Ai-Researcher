# -*- coding: utf-8 -*-
"""v2_llm.py - Shared OpenAI-compatible LLM transport (used by HERA and PACT).

v2.2.1 hardening: failures are NEVER silent. Every failed call logs the
reason to stderr (which lands in the task log), retries transient errors
(429/5xx/network) with backoff, and throttles concurrent calls from the
three parallel tasks via a host-wide file lock so dashscope rate limits
(429) cannot silently degrade HERA/PACT to deterministic fallbacks.

v2.2.1-rc4-hotfix (codegen latency, 20260806): codegen role (implementer
writes candidate code, child proposer drafts the next experiment) uses
STREAMING (SSE) so a slow-but-alive generation keeps flowing instead of
hitting a fixed total-request timeout: dashscope buffers the full
completion for non-streaming requests, so qwen3.8-max writing a full
15-19k-prompt script (measured 150s+) used to die at the read timeout.
Defaults: LLM_CODE_TIMEOUT=600s (per-chunk read), LLM_CODE_TOTAL=1200s
(overall cap), LLM_CODE_MAX_TOKENS=4000, LLM_CODE_ATTEMPTS=2. EVERY call
logs START / OK / retry / FAIL with sec and (for codegen) t_first, so a
child can never burn 10-30 minutes invisibly.

role=general (HERA decisions / repairs) keeps the buffered path
(300s, 3 attempts) - its calls complete in ~1-2 min.
"""
import json
import os
import sys
import time

try:
    import httpx  # module-level so _stream_completion can use it too
except Exception:  # noqa: BLE001 - environments without httpx fall back
    httpx = None


def _throttle(interval: float = 0.6) -> None:
    """Host-wide rate limiter: at most ~1 call per `interval` seconds.

    Uses a lock file so the three concurrent task processes share one
    throttle. Best-effort: on platforms without fcntl (Windows) it no-ops.
    """
    try:
        import fcntl
        with open("/tmp/v2_llm_ratelimit.lock", "w", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            time.sleep(interval)
            fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001 - throttling is best-effort
        time.sleep(min(interval, 0.2))


def _log(msg: str) -> None:
    try:
        print("[llm] %s" % msg, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default
    except (TypeError, ValueError):
        return default


def _role_config(role: str) -> dict:
    """Per-role timeout / token caps / attempts (env-overridable).

    role=codegen: implementer/proposer writes candidate code or a short
    next-experiment hypothesis. Streamed (SSE), 600s per-chunk read,
    1200s overall cap, 4000 output tokens, 2 attempts
    (LLM_CODE_TIMEOUT / LLM_CODE_TOTAL / LLM_CODE_MAX_TOKENS /
    LLM_CODE_ATTEMPTS overrides).
    role=general (HERA decisions / repairs): buffered, bounded default
    (300s, 3 attempts) - still visible via START/retry/FAIL lines.
    """
    role = str(role or "general").strip().lower()
    if role == "codegen":
        return {
            "timeout": _env_float("LLM_CODE_TIMEOUT", 600.0),
            "total": _env_float("LLM_CODE_TOTAL", 1200.0),
            "max_tokens": _env_int("LLM_CODE_MAX_TOKENS", 4000),
            "attempts": max(1, _env_int("LLM_CODE_ATTEMPTS", 2)),
        }
    return {
        "timeout": _env_float("LLM_TIMEOUT", 300.0),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 6000),
        "attempts": max(1, _env_int("LLM_ATTEMPTS", 3)),
    }


def _stream_completion(url, headers, body, timeout, total_cap):
    """SSE streaming chat completion. Returns (content, err, t_first).

    Streams delta chunks as they are generated, so a slow-but-alive
    model never dies on a fixed total-request timeout. A silent gap
    longer than `timeout`, OR total wall clock past `total_cap`
    (checked on every line), aborts the call.
    """
    t0 = time.time()
    first_token = None
    parts = []
    raw = []
    saw_sse = False
    with httpx.stream("POST", url, headers=headers, json=body,
                      timeout=httpx.Timeout(connect=30.0, read=timeout,
                                            write=60.0, pool=30.0)) as r:
        if r.status_code != 200:
            return None, ("status=%s body=%s"
                          % (r.status_code, (r.text or "")[:200])), None
        for line in r.iter_lines():
            if not line:
                continue
            # HARD wall-clock cap: checked on EVERY received line (not only
            # content chunks), so reasoning/keep-alive deltas cannot make a
            # call hang indefinitely.
            if total_cap and (time.time() - t0) > total_cap:
                return None, ("total>%.0fs cap" % total_cap), first_token
            raw.append(line)
            if not line.startswith("data:"):
                continue
            saw_sse = True
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:  # noqa: BLE001 - tolerate keep-alive noise
                continue
            try:
                ch = obj["choices"][0]
                delta = ch.get("delta") or ch.get("message") or {}
                text = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                continue
            if text:
                if first_token is None:
                    first_token = time.time() - t0
                parts.append(text)
                if total_cap and (time.time() - t0) > total_cap:
                    return None, ("total>%.0fs cap" % total_cap), first_token
    if not saw_sse and raw:
        # Some compatible endpoints answer with a single JSON body
        # even when stream=true was requested - parse it as fallback.
        try:
            obj = json.loads("\n".join(raw))
            content = obj["choices"][0]["message"]["content"]
            if content:
                return str(content), None, (time.time() - t0)
        except Exception:  # noqa: BLE001
            pass
    content = "".join(parts)
    if not content or not str(content).strip():
        return None, "empty streamed content", first_token
    return str(content), None, first_token


def default_llm_call(prompt: str, role: str = "general") -> str:
    """OpenAI-compatible chat completion. Returns '{}' when unavailable.

    role selects the timeout/token/attempt profile; codegen streams via
    SSE. Every call logs START and a duration line so LLM latency is
    observable in the run/daemon logs and a slow call is never silent.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("LLM_MODEL") or "gpt-4o"
    cfg = _role_config(role)
    timeout = float(cfg["timeout"])
    total_cap = float(cfg.get("total", 0.0) or 0.0)
    max_tokens = int(cfg["max_tokens"])
    attempts = int(cfg["attempts"])
    stream = role == "codegen"
    if not key:
        _log("no OPENAI_API_KEY; deterministic fallback")
        return "{}"
    if httpx is None:
        _log("httpx unavailable; deterministic fallback")
        return "{}"
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.5") or 0.5)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt[:int(
            os.environ.get("LLM_PROMPT_MAX", "32000") or 32000)]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
    url = base.rstrip("/") + "/chat/completions" if base else "https://api.openai.com/v1/chat/completions"
    last_err = ""
    for attempt in range(1, attempts + 1):
        _throttle()
        _log("START role=%s attempt=%d/%d timeout=%.0fs prompt_len=%d%s"
             % (role, attempt, attempts, timeout, len(prompt),
                " stream=1" if stream else ""))
        t0 = time.time()
        try:
            if stream:
                content, err, t_first = _stream_completion(
                    url, headers, body, timeout, total_cap)
                dt = time.time() - t0
                if content is not None:
                    _log("OK role=%s attempt=%d sec=%.1fs t_first=%.1fs "
                         "prompt_len=%d resp_len=%d"
                         % (role, attempt, dt, t_first or 0.0,
                            len(prompt), len(content)))
                    return content
                last_err = err or "stream failed"
            else:
                resp = httpx.post(
                    url, headers=headers, json=body,
                    timeout=httpx.Timeout(connect=30.0, read=timeout,
                                          write=60.0, pool=30.0))
                dt = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        _log("malformed 200 response: %s" % str(data)[:300])
                        return "{}"
                    if not content or not str(content).strip():
                        _log("empty content from %s" % model)
                        return "{}"
                    _log("OK role=%s attempt=%d sec=%.1fs prompt_len=%d "
                         "resp_len=%d" % (role, attempt, dt, len(prompt),
                                          len(str(content))))
                    return str(content)
                last_err = "status=%s body=%s" % (resp.status_code,
                                                  (resp.text or "")[:200])
            if last_err and attempt < attempts and any(
                    tag in last_err for tag in ("status=408", "status=429",
                                                "status=500", "status=502",
                                                "status=503", "status=504",
                                                "ReadTimeout", "ConnectError",
                                                "ConnectTimeout",
                                                "RemoteProtocolError",
                                                "ReadError")):
                _log("retry role=%s attempt=%d/%d sec=%.1fs err=%s"
                     % (role, attempt, attempts, time.time() - t0,
                        last_err[:160]))
                time.sleep(5 * attempt)
                continue
            _log("FAIL role=%s attempt=%d/%d sec=%.1fs err=%s prompt_len=%d"
                 % (role, attempt, attempts, time.time() - t0,
                    last_err[:160], len(prompt)))
            break
        except Exception as exc:  # noqa: BLE001 - retry transient failures
            dt = time.time() - t0
            last_err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            if attempt < attempts:
                _log("retry role=%s attempt=%d/%d sec=%.1fs err=%s"
                     % (role, attempt, attempts, dt, last_err[:160]))
                time.sleep(2 * attempt)
            else:
                _log("FAIL role=%s attempt=%d/%d sec=%.1fs err=%s prompt_len=%d"
                     % (role, attempt, attempts, dt, last_err[:160],
                        len(prompt)))
    _log("FAIL role=%s attempts=%d err=%s prompt_len=%d"
         % (role, attempts, last_err, len(prompt)))
    return "{}"


def codegen_llm_call(prompt: str) -> str:
    """Implementer/proposer role: streamed, bounded latency + tokens."""
    return default_llm_call(prompt, role="codegen")