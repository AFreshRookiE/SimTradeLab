# 🚀 ETFQuantDesk

**ETF 专属量化分析桌面软件**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-orange.svg)](#)

> 全流程: ETF 数据 → 因子生成 → 机器学习 → 回测验证

---

## 🎯 项目定位

ETFQuantDesk 是一款面向 ETF 的量化分析桌面软件，核心特色：

- **ETF 专属算子** — 溢价率(premium_rate)、跟踪误差(tracking_error)、IOPV偏离度(iopv_deviation)
- **自动因子挖掘** — 15个预置因子 + 自定义表达式，IC/ICIR 自动评估
- **ML 因子筛选** — IC过滤 + 去相关，XGBoost 预测模型训练
- **一键全流程** — 数据检查→因子生成→模型训练→回测验证，一键完成
- **暗色主题 UI** — NiceGUI + ECharts，GitHub 风格暗色界面

---

## 📸 功能截图

| 页面 | 功能 |
|------|------|
| 📊 数据管理 | ETF列表、分类筛选、K线走势、数据覆盖范围 |
| 🧬 因子管理 | 因子池、因子生成、因子筛选、模型训练、定时任务、算子参考 |
| 📝 策略编辑 | CodeMirror编辑器、4种模板策略、保存/加载 |
| 📈 回测执行 | 参数配置、一键运行、全流程、净值曲线、回撤图、交易明细 |
| 📊 结果展示 | 指标面板、净值曲线、回撤图、交易明细、CSV导出 |

---

## 🏗️ 架构

```
ETFQuantDesk
├── src/etfquant/
│   ├── core/          # 配置、日志
│   ├── data/          # DataBridge (Parquet读取+NAV/Premium合并)
│   ├── alpha/         # AlphaCalculator + FactorStore + AlphaScheduler
│   ├── ml/            # ModelTrainer + FactorScreener
│   ├── backtest/      # ETFBacktester (4策略模板)
│   ├── pipeline/      # PipelineBus (4阶段异步)
│   ├── api/           # DataService/FactorService/BacktestService/StrategyService
│   └── ui/            # NiceGUI 5页面 + ECharts暗色主题
├── config/            # YAML配置
├── strategies/        # 策略文件
└── output/            # 回测结果
```

**数据流:**

```
AKShare_Module (Parquet) → DataBridge → AlphaCalculator → FactorStore (SQLite)
                                                    ↓
                                              FactorScreener → ModelTrainer (XGBoost)
                                                    ↓
                                              ETFBacktester → ECharts 可视化
```

---

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/kay-ou/ETFQuantDesk.git
cd ETFQuantDesk

# 安装依赖
pip install -e .
```

**依赖:**
- Python >= 3.10
- nicegui >= 2.0
- pandas, numpy, xgboost, scikit-learn
- pyyaml, requests

**数据源:** 使用 [AKShare_Module](https://github.com/kay-ou/AKShare_Module) 导出的 Parquet 文件，放置于 `D:\AKShare_Module\data\export\` 目录。

---

## 🚀 启动

```bash
python run.py
```

浏览器打开 http://localhost:8080

---

## 🧬 ETF 专属算子

| 算子 | 说明 |
|------|------|
| `premium_rate()` | ETF溢价率 = (收盘价/单位净值 - 1) × 100% |
| `tracking_error(n)` | n日跟踪误差 |
| `iopv_deviation()` | IOPV偏离度 |

**通用算子:** `ts_return`, `ts_std`, `ts_mean`, `ts_max`, `ts_min`, `ts_rank`, `ts_corr`, `ts_delta`, `ts_sum`

---

## 📈 策略模板

| 策略 | 说明 |
|------|------|
| MA均线策略 | 双均线交叉 (5日/20日) |
| 动量策略 | 20日动量因子 |
| 均值回归策略 | 偏离均值反向交易 |
| ETF溢价策略 | 溢价率均值回归 |

---

## ⚙️ 配置

编辑 `config/etfquant.yaml`:

```yaml
data:
  parquet_dir: "D:/AKShare_Module/data/export"
  nav_dir: "D:/AKShare_Module/data/export/etf_nav"
  premium_dir: "D:/AKShare_Module/data/export/etf_premium"

alpha:
  schedule:
    enabled: true
    start_time: "18:00"
    end_time: "22:00"
  resources:
    gpu_utilization_limit: 0.8
    memory_limit_gb: 8.0

ml:
  predict_days: 5
  factor_screen:
    ic_threshold: 0.03
    icir_threshold: 0.5

backtest:
  benchmark: "510300.SH"
  initial_capital: 1000000
  t_plus_1: false    # ETF默认T+0

ui:
  host: "localhost"
  port: 8080
```

---

## 🔗 关联项目

| 项目 | 说明 |
|------|------|
| [AKShare_Module](https://github.com/kay-ou/AKShare_Module) | ETF数据仓库 (DolphinDB + Parquet导出) |
| [SimTradeLab](https://github.com/kay-ou/SimTradeLab) | 量化回测框架 (PTrade API模拟) |
| [SimTradeDesk](https://github.com/kay-ou/SimTradeDesk) | 量化桌面软件 (SimTradeLab GUI) |
| [alphagen](https://github.com/kay-ou/alphagen) | 因子挖掘引擎 |
| [SimTradeML](https://github.com/kay-ou/SimTradeML) | 机器学习模块 |
| [ptradeAPI](https://github.com/kay-ou/ptradeAPI) | PTrade API 参考 |

---

## 📄 License

**AGPL-3.0** — 详见 [LICENSE](LICENSE)

---

## ⚖️ 免责声明

本软件仅供学习和研究使用，不构成任何投资建议。使用者应自行承担投资风险，并遵守当地法律法规。
