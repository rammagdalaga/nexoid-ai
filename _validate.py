import ast
import sys
import os

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
]

all_ok = True
for f in files:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
    if not os.path.exists(path):
        print(f'  ? {f} (missing)')
        all_ok = False
        continue
    try:
        with open(path, encoding='utf-8') as fh:
            ast.parse(fh.read())
        print(f'  ✓ {f}')
    except SyntaxError as e:
        print(f'  ✗ {f}: {e}')
        all_ok = False

print(f'\n{"All files OK!" if all_ok else "Some files have errors!"}')
sys.exit(0 if all_ok else 1)