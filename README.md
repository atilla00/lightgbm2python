# lightgbm2python
Export lightgbm model to numpy/scipy for inference. Supports classification/regression with categorical features & nan values (use python None while using exported code).

## Examples
```python
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lgbm_exporter

df = pd.read_csv(
    "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
)
cols = df.columns[:-1]
target = df.columns[-1]
num_classes = df[target].nunique()

X, y = df[cols], df[target]

model = LGBMClassifier(verbose=-1)
model.fit(X, y)


# Categorical features supported with pandas.Categorical, use get_category_mappings_from_X() for mappings
# categoric_mappings = lgbm_exporter.get_category_mappings_from_X(X)
lgbm_exporter.export_model(
    model, 
    #categoric_mappings={},
    output_path="lgbm_model_export.py"
)

import lgbm_model_export
idx = 3
example = X.iloc[idx, :].replace(np.nan, None).to_dict()
print(example)
print(model.predict_proba(X)[idx])
print(lgbm_model_export.predict(**example))
```
