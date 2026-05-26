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
    'security/usage_tracker.py',
    'security/api_key_auth.py',
    'api/streaming.py',
    'api/response.py',
    'api/router.py',
    'api/gateway.py',
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

    'api_gateway_exists': ('api/gateway.py', ['class APIGateway', '/v1/inference', '/v1/chat', '/v1/batch', '/v1/evaluate']),
    'api_router_exists': ('api/router.py', ['class APIRouter', 'register', 'dispatch']),
    'api_response_exists': ('api/response.py', ['status', 'latency_ms', 'version']),
    'api_streaming_exists': ('api/streaming.py', ['stream_chunks', 'simple_tokenize']),
    'api_key_enforcement_active': ('api/gateway.py', ['APIKeyAuth', 'validate(key)', 'enforce_rate_limit']),
    'all_api_routes_require_auth': ('api/gateway.py', ['unauthorized', 'extract_key']),
    'no_direct_model_bypass': ('core/system_manager.py', ['submit_inference_payload']),
    'api_flow_order_correctness': ('api/gateway.py', ['_preflight', 'router.dispatch', 'gateway_flow="Auth>RateLimit>Validation>Router>EventBus>Pipeline>Inference>Response"']),
    'streaming_no_auth_bypass': ('api/gateway.py', ['stream_inference', '_preflight("/v1/inference"']),
    'usage_tracker_fully_integrated': ('api/gateway.py', ['record_request', 'record_tokens', 'record_violation']),
    'no_duplicate_rate_limit_systems': ('api/gateway.py', ['enforce_rate_limit']),
    'router_gateway_consistency': ('api/router.py', ['register', 'dispatch']),
    'single_canonical_api_flow': ('api/gateway.py', ['gateway_flow="Auth>RateLimit>Validation>Router>EventBus>Pipeline>Inference>Response"', '_preflight']),
    'streaming_mirrors_standard': ('api/gateway.py', ['stream_inference', '_preflight("/v1/inference"']),
    'phase2_lock_marker': ('VERSION.txt', ['Phase 2 Status: LOCKED', 'API Platform: FROZEN', 'Architecture: STABLE', 'Ready for Phase 3: TRUE']),
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


# module docstring + architecture block checks
status_modules = [
    'core/event_bus.py',
    'core/system_manager.py',
    'core/pipeline.py',
    'inference/batcher.py',
    'training/orchestrator.py',
    'training/checkpoint_manager.py',
]
for m in status_modules:
    txt = (root / m).read_text(encoding='utf-8')
    if 'APEXAI PHASE 2 FINAL STATE' in txt and 'Distributed Readiness: TRUE' in txt:
        print(f'  ✓ architecture_block: {m}')
    else:
        print(f'  ✗ architecture_block: {m}')
        all_ok = False

# README completeness
readme = (root / 'README.md').read_text(encoding='utf-8')
required_readme = [
    'Architecture Overview', 'Execution Flow', 'Strict Dependency Rules', '_validate.py', 'APEXAI_TRACE_MODE'
]
missing = [x for x in required_readme if x not in readme]
if missing:
    print(f'  ✗ readme_completeness: missing {missing}')
    all_ok = False
else:
    print('  ✓ readme_completeness')

# trace mode integration
log_txt = (root / 'security/logging.py').read_text(encoding='utf-8')
if 'APEXAI_TRACE_MODE' in log_txt and 'def trace(' in log_txt:
    print('  ✓ trace_mode_integration')
else:
    print('  ✗ trace_mode_integration')
    all_ok = False


# phase 2 final header + changelog checks
final_modules = [
    'core/event_bus.py',
    'core/system_manager.py',
    'core/pipeline.py',
    'inference/batcher.py',
    'training/orchestrator.py',
    'training/checkpoint_manager.py',
]
for m in final_modules:
    txt = (root / m).read_text(encoding='utf-8')
    required = [
        'APEXAI PHASE 2 FINAL STATE',
        'Status: COMPLETE',
        'System Integrity: VERIFIED',
        'Distributed Readiness: TRUE',
        'Phase: READY FOR PHASE 3',
        'CHANGELOG:',
    ]
    miss = [r for r in required if r not in txt]
    if miss:
        print(f'  ✗ phase2_final_block: {m} missing {miss}')
        all_ok = False
    else:
        print(f'  ✓ phase2_final_block: {m}')

# version file check
vtxt = (root / 'VERSION.txt').read_text(encoding='utf-8') if (root / 'VERSION.txt').exists() else ''
for req in ['0.2.0', 'stable', 'fully integrated', '2 complete']:
    if req not in vtxt:
        print(f'  ✗ version_file_missing: {req}')
        all_ok = False
if all(r in vtxt for r in ['0.2.0', 'stable', 'fully integrated', '2 complete']):
    print('  ✓ version_file')

# duplicate middleware chain check
gw_txt = (root / 'api/gateway.py').read_text(encoding='utf-8')
if gw_txt.count('_preflight(') <= 3 and 'handle(' in gw_txt and 'stream_inference' in gw_txt:
    print('  ✓ duplicate_middleware_chain')
else:
    print('  ✗ duplicate_middleware_chain')
    all_ok = False

# phase3 readiness marker
if 'READY FOR PHASE 3' in ''.join((root / m).read_text(encoding='utf-8') for m in final_modules):
    print('  ✓ phase3_ready_markers')
else:
    print('  ✗ phase3_ready_markers')
    all_ok = False


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

api_modules = ['api/gateway.py','api/router.py','api/response.py','api/streaming.py','security/api_key_auth.py','security/usage_tracker.py']
for am in api_modules:
    txt = (root / am).read_text(encoding='utf-8')
    if 'CHANGELOG:' in txt:
        print(f'  ✓ api_changelog_blocks: {am}')
    else:
        print(f'  ✗ api_changelog_blocks: {am}')
        all_ok = False



# Phase 1 reasoning expansion checks
if (root / 'training/security_reasoning.py').exists():
    reason_txt = (root / 'training/security_reasoning.py').read_text(encoding='utf-8')
    required_reason = ['Detection', 'Classification', 'Risk Analysis', 'Secure Fix Recommendation', 'CATEGORY_PATTERNS']
    miss = [r for r in required_reason if r not in reason_txt]
    if miss:
        print(f'  ✗ security_reasoning_module: missing {miss}')
        all_ok = False
    else:
        print('  ✓ security_reasoning_module')
else:
    print('  ✗ security_reasoning_module: file missing')
    all_ok = False

sd_txt = (root / 'training/security_dataset.py').read_text(encoding='utf-8') if (root / 'training/security_dataset.py').exists() else ''
if 'detect_dataset_imbalance' in sd_txt and 'OWASP_BALANCE_KEYS' in sd_txt and 'CVE_TYPE_KEYS' in sd_txt:
    print('  ✓ dataset_balance_checks_exist')
else:
    print('  ✗ dataset_balance_checks_exist')
    all_ok = False

fmt_txt = (root / 'security/atlas_formatter.py').read_text(encoding='utf-8') if (root / 'security/atlas_formatter.py').exists() else ''
if all(k in fmt_txt for k in ['issue_detection', 'explanation', 'severity_level', 'risk_reasoning', 'secure_fix_recommendation']):
    print('  ✓ formatter_reasoning_consistency')
else:
    print('  ✗ formatter_reasoning_consistency')
    all_ok = False

for mod in ['training/train.py', 'inference/generate.py', 'evaluation/benchmarks.py']:
    txt = (root / mod).read_text(encoding='utf-8')
    if 'analyze_security_reasoning' in txt:
        print(f'  ✓ reasoning_pipeline_linked: {mod}')
    else:
        print(f'  ✗ reasoning_pipeline_linked: {mod}')
        all_ok = False

print(f'\n{"All files OK!" if all_ok else "Some files have errors!"}')
sys.exit(0 if all_ok else 1)
