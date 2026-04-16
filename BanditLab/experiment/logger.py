import json
import os
import csv
import numpy as np

class Logger:
    def __init__(self, output_file: str):
        self.output_file = output_file
        
    def log(self, data: dict):
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        
        def numpy_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            raise TypeError
            
        if self.output_file.endswith(".json"):
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, default=numpy_default)
        elif self.output_file.endswith(".csv"):
            if not data:
                return
            # Separate scalar-per-step data from nested/array data
            flat_keys = []
            nested_keys = []
            for k, v in data.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    first = v[0] if isinstance(v, list) else v.flat[0]
                    if isinstance(first, (list, np.ndarray)):
                        nested_keys.append(k)
                    else:
                        flat_keys.append(k)
                else:
                    flat_keys.append(k)
            if flat_keys:
                length = len(data[flat_keys[0]])
                with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(flat_keys)
                    for i in range(length):
                        row = [data[k][i] for k in flat_keys]
                        writer.writerow(row)
            # Write nested data (raw runs) as separate JSON sidecar
            if nested_keys:
                sidecar = self.output_file.replace('.csv', '_raw.json')
                nested_data = {k: data[k] for k in nested_keys}
                with open(sidecar, 'w', encoding='utf-8') as f:
                    json.dump(nested_data, f, indent=2, default=numpy_default)
