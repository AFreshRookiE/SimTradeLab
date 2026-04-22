from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etfquant.core.config import MLConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["ETFDataSource", "FeatureEngineer", "ModelTrainer", "ModelPackage"]

logger = get_logger("etfquant.ml")


class ETFDataSource:
    def __init__(self, data_bridge: DataBridge, config: MLConfig) -> None:
        self._bridge = data_bridge
        self._config = config

    def get_stock_list(self) -> list[str]:
        return self._bridge.list_etf_codes()

    def get_price_data(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        df = self._bridge.load_etf_daily(code)
        if df.empty:
            return df
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df

    def get_nav_data(self, code: str) -> pd.DataFrame:
        return self._bridge.load_etf_nav(code)

    def get_premium_data(self, code: str) -> pd.DataFrame:
        return self._bridge.load_etf_premium(code)

    def get_tracking_error(self, code: str, window: int = 20) -> pd.Series:
        return self._bridge.calc_tracking_error(code, window)


class FeatureEngineer:
    def __init__(self, data_source: ETFDataSource, config: MLConfig,
                 factor_expressions: list[dict[str, Any]] | None = None) -> None:
        self._ds = data_source
        self._config = config
        self._factor_expressions = factor_expressions

    def build_features(self, code: str) -> pd.DataFrame | None:
        price_df = self._ds.get_price_data(code)
        if price_df.empty or len(price_df) < self._config.lookback_days + 10:
            return None
        if self._factor_expressions:
            return self._build_features_from_expressions(code, price_df)
        return self._build_features_hardcoded(code, price_df)

    def _build_features_from_expressions(self, code: str, price_df: pd.DataFrame) -> pd.DataFrame | None:
        from etfquant.alpha.calculator import ETFAlphaCalculator
        from etfquant.core.config import AlphaConfig

        features = pd.DataFrame(index=price_df.index)
        close = price_df["close"].astype(float) if "close" in price_df.columns else price_df.iloc[:, 0].astype(float)

        alpha_cfg = AlphaConfig(max_etf_for_ic=200)
        calc = ETFAlphaCalculator(self._ds._bridge, alpha_cfg)

        for f_info in self._factor_expressions:
            expr = f_info.get("expression", "")
            name = f_info.get("name", "")
            if not expr:
                continue
            try:
                vals = calc._evaluate_expression(expr, code)
                if vals is not None and len(vals) == len(features.index):
                    col_name = name if name else expr[:30]
                    features[col_name] = vals.reindex(features.index)
            except Exception as exc:
                logger.warning("因子表达式求值失败: %s, code=%s, error=%s", expr, code, exc)

        features["target"] = close.pct_change(self._config.predict_days).shift(-self._config.predict_days)
        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    def _build_features_hardcoded(self, code: str, price_df: pd.DataFrame) -> pd.DataFrame | None:
        features = pd.DataFrame(index=price_df.index)
        close = price_df["close"].astype(float) if "close" in price_df.columns else price_df.iloc[:, 0].astype(float)
        open_ = price_df.get("open", close).astype(float)
        high = price_df.get("high", close).astype(float)
        low = price_df.get("low", close).astype(float)
        volume = price_df.get("volume", pd.Series(0, index=price_df.index)).astype(float)
        amount = price_df.get("amount", pd.Series(0, index=price_df.index)).astype(float)

        ret = close.pct_change()
        features["return_1d"] = ret
        features["return_5d"] = close.pct_change(5)
        features["return_10d"] = close.pct_change(10)
        features["return_20d"] = close.pct_change(20)

        for w in [5, 10, 20, 60]:
            features[f"ma_{w}"] = close.rolling(w).mean()
            features[f"ma_ratio_{w}"] = close / close.rolling(w).mean()
            features[f"vol_{w}"] = ret.rolling(w).std()
            features[f"turnover_{w}"] = volume.rolling(w).mean()

        features["high_low_ratio"] = (high - low) / close
        features["close_open_ratio"] = (close - open_) / open_
        features["upper_shadow"] = (high - np.maximum(close, open_)) / close
        features["lower_shadow"] = (np.minimum(close, open_) - low) / close

        features["volume_ratio_5_20"] = volume.rolling(5).mean() / volume.rolling(20).mean()
        features["price_volume_corr"] = ret.rolling(20).corr(volume.pct_change())

        nav_df = self._ds.get_nav_data(code)
        nav_col = None
        for c in ("nav", "单位净值", "nav_per_unit"):
            if c in nav_df.columns:
                nav_col = c
                break
        if nav_col is not None:
            nav = nav_df[nav_col].astype(float)
            merged_nav = nav.reindex(price_df.index, method="ffill")
            features["premium_rate"] = (close - merged_nav) / merged_nav
            features["nav_momentum_5d"] = merged_nav.pct_change(5)
            features["nav_momentum_20d"] = merged_nav.pct_change(20)

        te = self._ds.get_tracking_error(code)
        if not te.empty:
            features["tracking_error"] = te.reindex(price_df.index, method="ffill")

        features["target"] = close.pct_change(self._config.predict_days).shift(-self._config.predict_days)
        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    def build_dataset(self, codes: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if codes is None:
            codes = self._ds.get_stock_list()
        all_features: list[pd.DataFrame] = []
        for code in codes:
            feat = self.build_features(code)
            if feat is not None and not feat.empty:
                feat["code"] = code
                all_features.append(feat)
        if not all_features:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        combined = pd.concat(all_features, axis=0)
        combined = combined.dropna(subset=["target"])
        feature_cols = [c for c in combined.columns if c not in ("target", "code")]
        X = combined[feature_cols]
        y = combined["target"]
        dates = combined.index.to_series()
        return X, y, dates


@dataclass
class ModelPackage:
    model: Any = None
    scaler: Any = None
    metadata: dict[str, Any] = None
    feature_names: list[str] = None
    factor_expressions: list[dict[str, Any]] = None

    def save(self, path: str) -> None:
        import pickle

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "metadata": self.metadata or {},
            "feature_names": self.feature_names or [],
            "factor_expressions": self.factor_expressions or [],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("模型已保存: %s", path)

    @classmethod
    def load(cls, path: str) -> ModelPackage:
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            model=data["model"],
            scaler=data.get("scaler"),
            metadata=data.get("metadata", {}),
            feature_names=data.get("feature_names", []),
            factor_expressions=data.get("factor_expressions", []),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_names] if self.feature_names else features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)


class ModelTrainer:
    def __init__(self, config: MLConfig) -> None:
        self._config = config

    def train(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
              factor_expressions: list[dict[str, Any]] | None = None) -> ModelPackage:
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            StandardScaler = None

        try:
            import xgboost as xgb

            model_cls = xgb.XGBRegressor
            is_xgb = True
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor

                model_cls = HistGradientBoostingRegressor
                is_xgb = False
                logger.warning("xgboost 不可用，使用 HistGradientBoostingRegressor 替代（原生支持NaN）")
            except ImportError:
                raise ImportError("训练模型需要 xgboost 或 scikit-learn，请安装其一: pip install xgboost 或 pip install scikit-learn")

        sorted_idx = dates.argsort()
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]

        n = len(X_sorted)
        train_end = int(n * self._config.train_ratio)
        val_end = int(n * (self._config.train_ratio + self._config.val_ratio))

        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        X_val = X_sorted.iloc[train_end:val_end]
        y_val = y_sorted.iloc[train_end:val_end]

        if StandardScaler is not None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        else:
            scaler = None
            X_train_scaled = X_train.fillna(0).values
            X_val_scaled = X_val.fillna(0).values

        params = self._config.model_params.copy()
        if is_xgb:
            model = model_cls(**params)
        else:
            sklearn_params = {k: v for k, v in params.items() if k in ("max_depth", "learning_rate", "max_iter", "l2_regularization")}
            if "n_estimators" in params and "max_iter" not in sklearn_params:
                sklearn_params["max_iter"] = params["n_estimators"]
            model = model_cls(**sklearn_params)

        if is_xgb:
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)

        feature_names = list(X.columns)
        metadata = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "feature_count": len(feature_names),
            "train_period": f"{X_sorted.index[0]} ~ {X_sorted.index[train_end - 1]}",
            "val_period": f"{X_sorted.index[train_end]} ~ {X_sorted.index[val_end - 1]}",
        }

        pkg = ModelPackage(
            model=model,
            scaler=scaler,
            metadata=metadata,
            feature_names=feature_names,
            factor_expressions=factor_expressions,
        )
        logger.info("模型训练完成: %d 训练样本, %d 验证样本, %d 特征", len(X_train), len(X_val), len(feature_names))
        return pkg
