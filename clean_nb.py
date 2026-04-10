import json
filepath = 'Notebooks/05_Evaluation.ipynb'
with open(filepath, 'r') as f:
    nb = json.load(f)
nb['cells'] = [c for c in nb['cells'] if 'garch_all_preds.pkl' not in ''.join(c.get('source', []))]
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None
with open(filepath, 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook Cleaned!")
