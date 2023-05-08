# LegateBoost

GBM implementation on Legate. In development.

## Example

Run with the legate launcher
```bash
legate example_script.py
```

```python
import cunumeric as cn
import legateboost as lbst

X = cn.random.random((1000, 10))
y = cn.random.random(X.shape[0])
model = lbst.LBRegressor(verbose=1, n_estimators=100, random_state=0, max_depth=2).fit(
    X, y
)
```
## Installation

Dependencies:
- cunumeric
- sklearn

From the project directory
```
pip install .
```

For editable installation
```
pip install . -e
```

## Running tests
```
pytest legateboost/test
```

## Pre-commit hooks

The pre-commit package is used for linting, formatting and type checks. This project uses strict mypy type checking.

Install pre-commit.
```
pip install pre-commit
```
Run all checks manually.
```
pre-commit run --all-files
```
