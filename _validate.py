import ast
import sys
from pathlib import Path

files = [
    'models/config.py',
    'models/transformer.py',
    'training/train.py',
    'training/dataset.py',
    'training/optimizer.py',
    'training/distributed.py',
    'training/deepspeed_config.py',
    'training/seo_processor.py',
    'training/multilingual_aligner.py',
    'training/repo_parser.py',
    'training/profiler.py',
    'training/checkpoint_manager.py',
    'training/orchestrator.py',
    'inference/generate.py',
    'inference/streaming.py',
    'inference/server.py',
    'inference/batcher.py',
    'evaluation/__init__.py',
    'evaluation/benchmarks.py',
    'evaluation/seo_eval.py',
    'evaluation/humaneval.py',
    'system/health_monitor.py',
    'core/event_bus.py',
    'core/system_manager.py',
    'core/pipeline.py',
    'utils/memory.py',
    'utils/helpers.py',
    'tokenizer/tokenizer.py',
    'main.py',
    'security/rate_limit.py',
    'security/validation.py',
    'security/logging.py',
    'security/audit_scanner.py',
    'db/interface.py',
    'db/memory_db.py',
    'db/schema.py',
]

all_ok = True
root = Path(__file__).resolve().parent
for f in files:
    path = root / f
    if not path.exists():
        print(f'  ? {f} (missing)')
        all_ok = False
        continue
    try:
        ast.parse(path.read_text(encoding='utf-8'))
        print(f'  ✓ {f}')
    except SyntaxError as e:
        print(f'  ✗ {f}: {e}')
        all_ok = False

checks = {
    'system_manager_integrates_modules': (
        'core/system_manager.py',
        ['TrainingOrchestrator', 'CheckpointManager', 'InferenceBatcher', 'HealthMonitor', 'EventBus']
    ),
    'pipeline_flow_integrity': (
        'core/pipeline.py',
        ['dataset', 'training', 'checkpoint', 'inference', 'evaluation', 'run_full_pipeline']
    ),
    'event_bus_exists_and_used': (
        'core/system_manager.py',
        ['event_bus.publish', 'training_started', 'training_failed', 'checkpoint_saved', 'inference_batch_processed', 'security_violation_detected']
    ),
    'stateless_server': ('inference/server.py', ['create_rate_limiter']),
    'distributed_rate_limiter': ('security/rate_limit.py', ['RateLimiterBackend', 'RedisRateLimiter', 'SimulatedRedisRateLimiter']),
    'security_scanner': ('security/audit_scanner.py', ['scan_repo', 'hardcoded_secret']),
    'logging_pipeline': ('security/logging.py', ['StructuredLogger', 'FileRouter', 'CloudStubRouter']),
    'concurrency_safety_event_bus': ('core/event_bus.py', ['threading.RLock', '_event_seq', 'set_error_handler']),
    'concurrency_safety_batcher': ('inference/batcher.py', ['threading.RLock', '_inflight', 'batcher_shutdown']),
    'concurrency_safety_training': ('training/orchestrator.py', ['threading.RLock', '_worker_threads']),
    'concurrency_safety_checkpoint': ('training/checkpoint_manager.py', ['threading.RLock', 'os.replace']),
    'state_consistency_enforcement': ('core/system_manager.py', ['reconcile_state', '_reconcile_loop', 'state_reconciled']),
    'error_propagation_unified': ('core/system_manager.py', ['_emit_error', 'system_error', 'recovery']),
    'deterministic_pipeline': ('core/pipeline.py', ['run_id', '_valid_transition', 'run_full_pipeline']),
    'memory_safety_guards': ('utils/memory.py', ['trim_kv_cache', 'sliding_window_context', 'memory_pressure_detected']),
}

for name, (rel, needles) in checks.items():
    txt = (root / rel).read_text(encoding='utf-8') if (root / rel).exists() else ''
    missing = [n for n in needles if n not in txt]
    if missing:
        print(f'  ✗ {name}: missing {missing}')
        all_ok = False
    else:
        print(f'  ✓ {name}')

# no orphan modules check: all core modules must be referenced by system manager/pipeline
sys_txt = (root / 'core/system_manager.py').read_text(encoding='utf-8')
pipe_txt = (root / 'core/pipeline.py').read_text(encoding='utf-8')
orphans = []
if 'TrainingOrchestrator' not in sys_txt:
    orphans.append('training/orchestrator.py')
if 'CheckpointManager' not in sys_txt:
    orphans.append('training/checkpoint_manager.py')
if 'InferenceBatcher' not in sys_txt:
    orphans.append('inference/batcher.py')
if 'HealthMonitor' not in sys_txt:
    orphans.append('system/health_monitor.py')
if 'run_evaluation_stage' not in pipe_txt:
    orphans.append('evaluation/*')
if orphans:
    print(f'  ✗ orphan_modules: {orphans}')
    all_ok = False
else:
    print('  ✓ orphan_modules')

# forbidden dependencies usage check
forbidden = ['openai', 'anthropic', 'cohere', 'langchain', 'vllm', 'transformers', 'peft', 'trl', 'accelerate']
violations = []
for py in root.rglob('*.py'):
    if '__pycache__' in py.parts:
        continue
    txt = py.read_text(encoding='utf-8', errors='ignore').lower()
    for dep in forbidden:
        if f'import {dep}' in txt or f'from {dep} import' in txt:
            violations.append((str(py.relative_to(root)), dep))
if violations:
    for v in violations:
        print(f'  ✗ forbidden dependency: {v[0]} uses {v[1]}')
    all_ok = False
else:
    print('  ✓ forbidden dependency usage check')

# MoE integrity
moe_txt = (root / 'models/transformer.py').read_text(encoding='utf-8')
if 'Mixture of Experts (MoE) support' in moe_txt and 'class MoEFFN' in moe_txt:
    print('  ✓ moe_intact')
else:
    print('  ✗ moe_intact')
    all_ok = False

print(f'\n{"All files OK!" if all_ok else "Some files have errors!"}')
sys.exit(0 if all_ok else 1)

# orphan async tasks running untracked check
sm_txt = (root / 'core/system_manager.py').read_text(encoding='utf-8')
if '_reconcile_thread' in sm_txt and 'join(timeout=' in sm_txt:
    print('  ✓ orphan_async_tasks_tracked')
else:
    print('  ✗ orphan_async_tasks_tracked')
    all_ok = False

# event bus integrity under load markers
eb_txt = (root / 'core/event_bus.py').read_text(encoding='utf-8')
if '_event_seq' in eb_txt and '_history' in eb_txt and 'publish(' in eb_txt:
    print('  ✓ event_bus_integrity_markers')
else:
    print('  ✗ event_bus_integrity_markers')
    all_ok = False
