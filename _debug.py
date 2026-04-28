import sys
sys.path.insert(0, "src")
from etfquant.core.config import ETFQuantConfig
from etfquant.api.factor import FactorService

config = ETFQuantConfig()
svc = FactorService(config.alpha, config.data, config.ml)

all_factors = svc._store.list_all()
valid_factors = [f for f in all_factors if f.get("is_valid")]
print(f"Valid factors: {len(valid_factors)}")

groups = svc.find_redundant_factors(corr_threshold=0.95)
print(f"Groups found: {len(groups)}")
for g in groups:
    print(f"  Keep: {g['keep']} (IC={g['keep_ic']:.4f}), Remove: {g['remove_count']}, MaxCorr: {g['max_corr']:.4f}")
    for r in g['remove'][:3]:
        print(f"    - {r}")
