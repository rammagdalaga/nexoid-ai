![Nexoid Atlas Banner](./nexoid-banner.jpg)

# ApexAI — Phase 2 Finalized Runtime Core

ApexAI is a **from-scratch transformer intelligence system** focused on coding workflows.

## Phase 2 Status

Phase 2 hardening/finalization is complete:
- Concurrency-safe EventBus
- Deterministic, bounded InferenceBatcher
- Lifecycle-safe TrainingOrchestrator
- Atomic, integrity-verified CheckpointManager
- State-reconciled SystemManager
- Deterministic validated Pipeline
- Extended security and validation controls

## Architecture Overview

### Core Integration Modules
- `core/system_manager.py` — runtime lifecycle authority and subsystem coordinator.
- `core/pipeline.py` — deterministic stage flow + validated handoff envelopes.
- `core/event_bus.py` — internal publish/subscribe event transport.

### Runtime Subsystems
- `training/orchestrator.py` — training queue/worker lifecycle simulation.
- `training/checkpoint_manager.py` — versioned checkpoint save/load/verify/rotate.
- `inference/batcher.py` — bounded fair batch scheduling for inference requests.
- `system/health_monitor.py` — runtime health telemetry snapshots.
- `security/validation.py` / `security/rate_limit.py` / `security/logging.py` — input hardening, throttling, structured logging.

## Execution Flow (Text Diagram)

`dataset -> training -> checkpoint -> inference -> evaluation`

Implemented in `core/pipeline.py` via strict stage ordering and validation gates.

## Strict Dependency Rules

ApexAI enforces:
- **NO** OpenAI/Anthropic/Cohere SDK usage
- **NO** transformers/vLLM/LangChain/PEFT/TRL/accelerate usage
- Only HuggingFace `datasets` allowed for dataset ingestion

## Validation

Run repository validation:

```bash
python _validate.py
```

This checks syntax, module integration, security constraints, concurrency markers, deterministic pipeline markers, and forbidden dependencies.

## Training Entry Points

### Standard training CLI
```bash
python main.py train --config small
```

### Orchestrator-managed training lifecycle (integration path)
- Instantiate `SystemManager`
- Submit job with `submit_training_job(...)`
- Monitor using `check_training_job(...)`

## Trace Logging Mode

Structured trace mode is available via environment variable:

```bash
export APEXAI_TRACE_MODE=1
```

When enabled, trace-level events are emitted for pipeline/reconciliation and key lifecycle operations through existing structured logging routes.

---

Built under the Nexoid.ai / ApexAI system.


## Phase 2 Release Notes

Completed systems in Phase 2 final release:
- Security-hardened inference server with rate limiting and validation
- Distributed-safe orchestration and checkpoint integrity controls
- Deterministic pipeline + system manager reconciliation
- Concurrency-safe event bus and bounded batching runtime
- Structured logging with optional trace mode

Final system guarantees:
- No external AI APIs are used in runtime paths
- Stateless architecture is enforced in service behavior
- Distributed-safe design is implemented for orchestration and throttling backends


"""
CHANGELOG:
- Phase 2 final cleanup applied
- System integration verified
- Documentation standardized
- Ready for Phase 3 transition
"""


## API Platform Stabilization Summary

Stabilization actions completed:
- Unified API flow order in gateway: Auth -> Rate Limit -> Validation -> Router -> EventBus -> Pipeline -> Inference -> Response.
- Standardized error handling to normalized response envelopes (no stack trace exposure).
- Hardened streaming path to use gateway preflight (no auth/validation bypass) with partial-failure safety.
- Unified usage tracking and rate-limit violation accounting across normal and streaming paths.

Production readiness confirmation:
- API layer flow consistency verified
- Security enforcement unified
- Routing + streaming integration hardened
- System remains stateless and Phase 3 ready
