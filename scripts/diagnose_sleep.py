"""Offline, train-only diagnostics for the saved run1 adapters. No gate labels."""
import json
import argparse
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from baselines.sleep_hf import HFBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning', action='store_true', help='train-only 18/6 validation split')
    parser.add_argument('--baseline', action='store_true', help='add matched old-lr validation baseline')
    parser.add_argument('--steps', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1] / 'results' / 'smoke_sleep'
    saved = json.loads((root / 'run1' / 'evidence.json').read_text())
    backend = HFBackend(directory=root / 'diagnostics' / 'adapters')
    if args.learning or args.baseline:
        # Fix the split before seeing results. These are exclusively train items;
        # no hidden/gate/canary tasks or labels enter model selection.
        examples = saved['training'][0]['examples']
        train, validation = examples[:18], examples[18:]
        backend.lr = 4e-4
        original = torch.optim.AdamW
        probes = []

        class ProbedAdamW(original):
            completed = 0

            def step(self, *a, **kw):
                result = super().step(*a, **kw)
                self.completed += 1
                if self.completed in (64, 128, 256):
                    backend._cache.clear()
                    groups = {}
                    for name, items in [('train', train), ('validation', validation)]:
                        rows = [dict(prompt=p, target=t, answer=backend.generate(p, 'expert-0'))
                                for p, t in items]
                        groups[name] = dict(exact=sum(r['answer'].strip() == r['target'] for r in rows),
                                            total=len(rows), rows=rows)
                    probes.append(dict(steps=self.completed, **groups))
                    (root / 'learning_diagnosis.json').write_text(json.dumps(dict(
                        lr=backend.lr, seed=0, split='first 18 train; last 6 validation',
                        probes=probes), indent=2), encoding='utf-8')
                    print('Train-only probe:', self.completed,
                          {k: (v['exact'], v['total']) for k, v in groups.items()}, flush=True)
                    backend._cache.clear()
                    backend._activate('expert-0')
                return result

        if args.learning:
            torch.optim.AdamW = ProbedAdamW
            try:
                backend.train_expert(train, steps=args.steps, seed=0)
            finally:
                torch.optim.AdamW = original
        # Match the old run's learning rate on the identical train-only split.
        backend.lr = 2e-4
        baseline = backend.train_expert(train, steps=64, seed=0)
        rows = [dict(prompt=p, target=t, answer=backend.generate(p, baseline))
                for p, t in validation]
        path = root / 'learning_diagnosis.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['baseline'] = dict(lr=backend.lr, steps=64,
            validation=dict(exact=sum(r['answer'].strip() == r['target'] for r in rows),
                            total=len(rows), rows=rows))
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Baseline train-only validation:', payload['baseline']['validation']['exact'],
              '/', len(rows), flush=True)
        return
    for handle in ('expert-0', 'expert-2'):
        weights = torch.load(root / 'run1' / 'adapters' / (handle + '.pt'),
                             map_location='cpu', weights_only=True)
        backend.experts[handle] = {'finite': True, 'committed': True}
        for name, layer in backend.layers.items():
            layer.add(handle)
            with torch.no_grad():
                for k, p in layer.adapters[handle].items():
                    p.copy_(weights[name][k])
                    p.requires_grad_(False)
    examples = saved['training'][0]['examples']
    rows = []
    for prompt, target in examples:
        answer = backend.generate(prompt, 'expert-0')
        rows.append(dict(prompt=prompt, target=target, answer=answer,
                         target_tokens_with_eos=len(backend.tokenizer.encode(target,
                             add_special_tokens=False)) + 1,
                         generated_tokens=len(backend.tokenizer.encode(answer,
                             add_special_tokens=False)), exact=answer.strip() == target))
    probes = [p for p, _ in examples[:3]]
    damage = [dict(prompt=p, base=backend.generate(p),
                   damaged=backend.generate(p, 'expert-2')) for p in probes]
    payload = dict(train_fit=rows, damage_probes=damage,
                   norm_ratios={h: backend.adapter_delta_ratio(h) for h in backend.experts},
                   code_tokenization=[dict(code=c, tokens=backend.tokenizer.tokenize(c),
                       reversed_upper=c[::-1].upper(),
                       target_tokens=backend.tokenizer.tokenize(c[::-1].upper()))
                       for c in ('htqg', 'plpk', 'nkuh', 'ucpm', 'boxn')])
    (root / 'diagnosis.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved adapter ratios:', payload['norm_ratios'], flush=True)
    print('Old A train exact:', sum(r['exact'] for r in rows), '/', len(rows), flush=True)
    print('Target token range incl EOS:', min(r['target_tokens_with_eos'] for r in rows),
          max(r['target_tokens_with_eos'] for r in rows), flush=True)
    print('Damaged vs base outputs changed:', sum(r['base'] != r['damaged'] for r in damage),
          '/', len(damage), flush=True)


if __name__ == '__main__':
    main()
