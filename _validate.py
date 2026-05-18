import ast
import sys
import os
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
    'system/health_monitor.py',
    'inference/batcher.py',
    'training/checkpoint_manager.py',
    'training/orchestrator.py',
    'inference/generate.py',
    'inference/streaming.py',
    'inference/server.py',
    'evaluation/__init__.py',
    'evaluation/benchmarks.py',
    'evaluation/seo_eval.py',
    'evaluation/humaneval.py',
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
    'phase2_components': ('training/orchestrator.py', ['TrainingOrchestrator', 'JOB_COMPLETED']),
    'stateless_server': ('inference/server.py', ['stateless_mode=True', 'create_rate_limiter']),
    'distributed_rate_limiter': ('security/rate_limit.py', ['RateLimiterBackend', 'RedisRateLimiter', 'SimulatedRedisRateLimiter']),
    'security_scanner': ('security/audit_scanner.py', ['scan_repo', 'hardcoded_secret']),
    'logging_pipeline': ('security/logging.py', ['StructuredLogger', 'FileRouter', 'CloudStubRouter']),
}

for name, (rel, needles) in checks.items():
    txt = (root / rel).read_text(encoding='utf-8') if (root / rel).exists() else ''
    missing = [n for n in needles if n not in txt]
    if missing:
        print(f'  ✗ {name}: missing {missing}')
        all_ok = False
    else:
        print(f'  ✓ {name}')

# forbidden dependencies usage check
forbidden = ['openai', 'anthropic', 'cohere', 'langchain', 'vllm', 'transformers']
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

print(f'\n{"All files OK!" if all_ok else "Some files have errors!"}')
sys.exit(0 if all_ok else 1)


moe_txt = (root / "models/transformer.py").read_text(encoding="utf-8")
if "Mixture of Experts (MoE) support" in moe_txt and "class MoEFFN" in moe_txt:
    print("  ✓ moe_intact")
else:
    print("  ✗ moe_intact")
    all_ok = False
