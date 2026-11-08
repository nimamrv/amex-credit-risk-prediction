"""End-to-end smoke test using fictional customer data."""
import json
import subprocess
import sys
from pathlib import Path
from scripts.make_synthetic import generate


def test_train_smoke(tmp_path):
    csv = tmp_path / 'fictional.csv'
    out = tmp_path / 'outputs'
    generate(n_customers=60).to_csv(csv, index=False)
    result = subprocess.run([sys.executable, '-m', 'src.train', '--data', str(csv),
                             '--output', str(out), '--quick'],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    metrics = json.loads((out / 'metrics.json').read_text())
    assert 0 <= metrics['xgb_test_auc'] <= 1
    assert (out / 'xgb_model.joblib').exists()
