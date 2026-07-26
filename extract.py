import json

with open('Waxal_Challenge_Starter_Code.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('starter_code.py', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            out.write(''.join(cell['source']))
            out.write('\n\n')
