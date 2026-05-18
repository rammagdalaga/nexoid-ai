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
