# Alpha Automation

Conservative scaffold for a WorldQuant BRAIN alpha exploration service.

The service is designed around a hard submission gate:

```text
generated -> preflight_passed -> simulated -> metric_passed
-> check_pending | approved -> submitted
```

AI generation is never allowed to call submit directly. The worker can only submit through the guard after metrics and BRAIN submission checks pass.

## Safety Defaults

- `AUTO_SUBMIT=false` by default.
- Each worker round allows at most 4 final submits.
- Mandatory checks block submission unless they are final `PASS`.
- `PENDING`, `FAIL`, `ERROR`, and mandatory `WARNING` checks are not submitted.
- Self/prod correlation above `0.7` blocks submission even if a response is otherwise permissive.
- AI expressions are locally preflighted against an operator whitelist before simulation.
- OpenAI-compatible generation receives a research context before each call: local rules, generation patterns, candidate history, and optional reference data from the old Brain project.
- The first implementation uses deterministic local clients, not live BRAIN or AI APIs.

## Commands

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db init-db
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db status
PYTHONPATH=src python3 -m alpha.cli --db alpha.db fields --preset ind --limit 40
PYTHONPATH=src python3 -m alpha.cli --db alpha.db submit-approved
PYTHONPATH=src python3 -m alpha.cli --db alpha.db daemon --batch-size 8 --loop-seconds 60
PYTHONPATH=src python3 -m alpha.cli --db alpha.db web --host 0.0.0.0 --port 8080
```

The CLI reads `.env` by default and writes service logs to `logs/alpha.log`. Override with `--env-file` and `--log-file`.

To run a specific exploration scope without editing `.env`:

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset us-d0 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset chn-d0 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset eur-d1 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset glb --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset ind --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --region USA --universe TOP3000 --delay 0 --neutralization SUBINDUSTRY --decay 6 --truncation 0.03 --batch-size 1
```

Example `.env` for AI dry-run exploration:

```bash
AI_CLIENT=multi
AI_API_KEY=your_key_here
AI_BASE_URL=https://your-relay-host/v1
AI_MODEL_PROFILES=grok-4-1-fast-reasoning@controller,deepseek-v4-pro-free@critic,gemini-3-flash-free@generator,glm-4.5@optimizer,gpt-5.4-nano@validator
BRAIN_CLIENT=local
AUTO_SUBMIT=false
```

For separate relay tokens per model, use profile files instead of one shared key:

```bash
AI_CLIENT=multi
AI_MODEL_PROFILES_FILE=config/ai_model_profiles.json
BRAIN_CLIENT=http
AUTO_SUBMIT=false
```

Create the local files from the examples:

```bash
cp config/ai_model_profiles.example.json config/ai_model_profiles.json
cp secrets/ai/grok.env.example secrets/ai/grok.env
cp secrets/ai/gemini.env.example secrets/ai/gemini.env
cp secrets/ai/glm.env.example secrets/ai/glm.env
cp secrets/ai/nano.env.example secrets/ai/nano.env
```

Each `secrets/ai/*.env` file can hold a different relay token and URL:

```bash
AI_API_KEY=one_model_token_here
AI_BASE_URL=https://api.1314mc.net/v1
```

`config/ai_model_profiles.json` and `secrets/ai/*.env` are ignored by git. The example files are safe placeholders.

For live BRAIN dry-run checks, switch `BRAIN_CLIENT=http` and provide BRAIN credentials. Keep `AUTO_SUBMIT=false` until the logs prove the gate is behaving correctly.

```bash
BRAIN_CLIENT=http
BRAIN_EMAIL=your_brain_email
BRAIN_PASSWORD=your_brain_password
AUTO_SUBMIT=false
```

If you already have the old Brain credentials file, this also works:

```bash
BRAIN_CLIENT=http
BRAIN_CREDENTIALS_FILE=/root/brain_alpha/Brain/brain_credentials.txt
AUTO_SUBMIT=false
```

With `BRAIN_CLIENT=local`, no WorldQuant BRAIN backtest is performed. Local mode is only for wiring tests.

## Web Control Panel

Run the personal server control panel:

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db web --host 0.0.0.0 --port 8080
```

Open `http://SERVER_IP:8080`. The panel has no login layer. It can:

For the persistent server process on port `5000`, use the service helper:

```bash
scripts/alpha_web start
scripts/alpha_web status
scripts/alpha_web restart
scripts/alpha_web stop
```

On this server it is installed as `alpha-web.service`, with restart-on-failure and boot-time startup enabled. The persistent panel writes stdout/stderr to `logs/web.stdout.log` and is reachable at `http://SERVER_IP:5000`.

- choose `region`, `delay`, `universe`, and `neutralization` directly, with presets only used as shortcuts
- run each daemon round with batch size `8` by default and cap Web-launched batches at the BRAIN multisimulation limit of `8`
- start the daemon loop in the background
- pause the daemon with `SIGINT`, then `SIGTERM` if needed
- show candidate status counts and recent expressions from SQLite
- tail `logs/alpha.log` and `logs/daemon.stdout.log`
- inspect the active scope field pool on demand

The daemon launched by the panel is the same guarded CLI path:

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db daemon --region <region> --universe <universe> --delay <delay> --neutralization <neutralization> --batch-size 8 --loop-seconds <seconds>
```

Starting or stopping from the panel does not bypass field validation, BRAIN backtesting, retry limits, correlation checks, or the final submit cap.

## Configuration

Environment variables:

- `AUTO_SUBMIT`: defaults to `false`
- `AI_CLIENT`: defaults to `local`; use `openai` for an OpenAI-compatible chat completions API
- `AI_CLIENT=multi`: runs the multi-model orchestrator
- `AI_API_KEY` or `OPENAI_API_KEY`: required for `AI_CLIENT=openai` and shared-relay `AI_CLIENT=multi`
- `AI_BASE_URL`: defaults to `https://api.openai.com/v1`; relay URLs may be either `https://host/v1` or the full `https://host/v1/chat/completions`
- `AI_MODEL`: defaults to `gpt-4.1-mini`
- `AI_MODEL_PROFILES`: compact model role list for `AI_CLIENT=multi`; default is `grok-4-1-fast-reasoning@controller,deepseek-v4-pro-free@critic,gemini-3-flash-free@generator,glm-4.5@optimizer,gpt-5.4-nano@validator`
- `AI_MODEL_PROFILES_FILE`: optional JSON profile file for `AI_CLIENT=multi`; overrides `AI_MODEL_PROFILES`
- `env_file` in a JSON model profile: optional per-model env file containing `AI_API_KEY` / `OPENAI_API_KEY` and `AI_BASE_URL`
- `request_timeout` in a JSON model profile: optional per-model HTTP timeout in seconds; useful for slow free routes
- `BRAIN_CLIENT`: defaults to `local`; use `http` for live BRAIN HTTP API
- `BRAIN_EMAIL` / `BRAIN_PASSWORD`: optional live BRAIN authentication credentials
- `BRAIN_CREDENTIALS_FILE`: optional JSON credentials file compatible with the old Brain project format
- `BRAIN_BASE_URL`: defaults to `https://api.worldquantbrain.com`
- `BRAIN_DATAFIELD_RETRIES`: defaults to `2`
- `ALPHA_REGION`: defaults to `USA`
- `ALPHA_UNIVERSE`: defaults to `TOP3000`
- `ALPHA_DELAY`: defaults to `1`
- `ALPHA_NEUTRALIZATION`: defaults to `INDUSTRY`
- `ALPHA_DECAY`: defaults to `0`
- `ALPHA_TRUNCATION`: defaults to `0.05`
- `ALPHA_FIELD_DISCOVERY`: defaults to `true`; fetches BRAIN datafields for the active scope before AI generation
- `ALPHA_FIELD_LIMIT`: defaults to `120`
- `ALPHA_FIELD_SEARCHES`: optional comma-separated data-field search terms
- `ALPHA_FIELD_CACHE_DIR`: defaults to `data/field_cache`
- `ALPHA_FIELD_CACHE_TTL_SECONDS`: defaults to `86400`
- `ALPHA_KNOWLEDGE_DIR`: defaults to `knowledge/`; contains prompt knowledge files loaded before AI generation
- `REFERENCE_BRAIN_DIR`: defaults to `/root/brain_alpha/Brain`; if present, summarizes `submitted_alphas.csv`, `fail_alphas.csv`, and `templates_usa_d0_success_submitted.json`
- `MAX_FINAL_SUBMITS_PER_ROUND`: defaults to `4`
- `MAX_RETRIES`: defaults to `3`
- `MIN_SHARPE`: defaults to `1.58`
- `MIN_FITNESS`: defaults to `1.0`
- `MAX_CORRELATION`: defaults to `0.7`
- `ALPHA_DB`: defaults to `alpha.db`
- `BATCH_SIZE`: defaults to `8`
- `LOOP_SECONDS`: defaults to `60`

## Next Integration Points

Add live adapters behind `AIClient` and `BrainClient` in `src/alpha/clients.py`.
Keep `SubmissionPolicy` and `evaluate_submission_readiness` as the final gate before any live submit call.

## AI Research Context

Before each AI generation call, the worker builds `research_context` from:

- `knowledge/wqb_rules.md`
- `knowledge/generation_patterns.md`
- `knowledge/usa_d0_patterns.md`
- BRAIN datafields for the active `region/universe/delay`, cached locally and exposed as an allowed field pool
- recent local candidate failures, pending checks, and approved/submitted alphas from SQLite
- optional reference artifacts from `REFERENCE_BRAIN_DIR`

After each simulated candidate, the worker stores the alpha id, IS metrics, submission checks, generated hypothesis, risk notes, failure reasons, and status transitions in SQLite. The next AI call receives that feedback, so the model can avoid repeated failing structures and build on candidates that passed real platform checks.

The OpenAI-compatible adapter asks for research-grade candidates and filters obvious trivial outputs such as `rank(close)` when research mode is active. This is prompt/context learning, not model fine-tuning.

With `AI_CLIENT=multi`, the orchestrator uses five roles:

- `grok-4-1-fast-reasoning@controller`: reads the current experiment plan and drafts/finalizes generator guidance
- `deepseek-v4-pro-free@critic`: reviews the controller draft for repeated families, weak anchors, regional threshold misses, and insufficient G-1/G-2 separation
- `gemini-3-flash-free@generator`: broad candidate exploration
- `glm-4.5@optimizer`: optimization-heavy generation, especially around near-threshold anchors
- `gpt-5.4-nano@validator`: cheap JSON/format/field sanity pass before preflight and BRAIN simulation

Each generated candidate stores its model source, model role, model name, controller allocation, and validator metadata in SQLite events. The web panel shows per-model generated/approved/failed counts and best Sharpe/Fitness for the current run.

Field selection is service-owned. The AI does not call BRAIN directly and should not invent datafield ids. When live field discovery is available, the worker fetches `/data-fields` for the chosen scope, passes the resulting `field_ids` to the AI, and preflight rejects expressions containing unknown datafields before simulation.

Simulation scope is config-owned. The AI may return a `settings` object, but the service records it only as `proposed_settings`; it cannot override region, universe, delay, neutralization, decay, or truncation used for BRAIN simulation.

For quick scope switching, `run-once`, `daemon`, and `check-ai` accept:

- `--preset us-d0` / `--preset us-d1`
- `--preset chn-d0` / `--preset chn-d1`
- `--preset eur-d0` / `--preset eur-d1`
- `--preset glb`
- `--preset ind`
- `--preset asi`, `--preset hkg`, `--preset kor`
- `--region`, `--universe`, `--delay`, `--neutralization`, `--decay`, `--truncation`

Use `PYTHONPATH=src python3 -m alpha.cli presets` to print every built-in scope preset. Current platform-validated defaults use `IND` and `GLB` with `delay=1`; `CHN`, `EUR`, and `USA` have both `delay=0` and `delay=1` presets.

## Closed Loop Modes

`BRAIN_CLIENT=local`:

- Generates candidates.
- Uses a deterministic fake simulation.
- Useful only for testing the service code.

`BRAIN_CLIENT=http` and `AUTO_SUBMIT=false`:

- Generates candidates.
- Sends 2-8 expressions as one WorldQuant BRAIN multisimulation request to `/simulations`; single-candidate batches use the normal single simulation request.
- Polls the simulation URLs until alpha ids or platform errors are returned.
- Fetches `/alphas/{id}` and `/alphas/{id}/check`.
- Stores real metrics/checks in SQLite.
- Approves only candidates whose real checks pass the guard.
- Does not submit.

`BRAIN_CLIENT=http` and `AUTO_SUBMIT=true`:

- Runs the same real simulation and guard.
- Submits only after approval.
- Verifies the platform moved the alpha to `stage=OS` with `dateSubmitted`.
- Still respects the final-submit cap.

## Relay Endpoint Notes

For OpenAI-compatible relay services, set:

```bash
AI_CLIENT=openai
AI_API_KEY=your_relay_key
AI_BASE_URL=https://your-relay-host/v1
```

If your provider gives the full endpoint, this also works:

```bash
AI_BASE_URL=https://your-relay-host/v1/chat/completions
```

Do not set `AI_BASE_URL` to only `/v1/chat/completions`; it must include the scheme and host.
