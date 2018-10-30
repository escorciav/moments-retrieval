# a-c

Traditional ablation analysis:

Local+Global+TEF
Local+Global
Local

# d[0-9]

only TEF

## d1

The keys corresponding to the results were manually edited due to a humam error launching the experiment at 4AM. The snippet below details the change:

```python
import json

for file_index in range(1, 6):
    filename = f'data/interim/mcn_15/d1/{file_index}.json'
    with open(filename, 'r') as fid:
        data = json.load(fid)
        rm_elements = []
        for i in data:
            if i.startswith('test_r'):
                rm_elements.append(i)
        for i in rm_elements:
            data[i.replace('test', 'val')] = data[i]
            del data[i]
    with open(filename, 'w') as fid:
        json.dump(data, fid)
```