from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path

from nicegui import ui

from etfquant.core.logger import get_logger

logger = get_logger("etfquant.ui.factor_page")

from etfquant.core.config import ETFQuantConfig, ScheduleConfig
from etfquant.api.factor import FactorService
from etfquant.ml.factor_screener import FactorScreener
from etfquant.ui.echart_theme import ECHART_GRID, ECHART_XAXIS, ECHART_YAXIS

warnings.filterwarnings("ignore", category=RuntimeWarning)


_PARAM_HINTS = """
**参数说明与建议范围：**

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| IC阈值 | 0.02~0.05 | 因子IC超过此值才视为有效。0.03为常规标准，0.05为严格标准 |
| RankIC阈值 | 0.02~0.05 | 秩相关IC，对异常值更鲁棒，建议与IC阈值保持一致 |
| ICIR阈值 | 0.3~1.0 | IC的均值/标准差，衡量因子稳定性。>0.5为稳定，>1.0为非常稳定 |
| 预测周期 | 1~20 | 因子预测未来N天的收益。1=日频，5=周频，20=月频 |
| 最大ETF数(IC) | 50~1035 | 计算因子IC时抽样多少只ETF。越多IC越稳定准确，但耗时增加 |
| 最大ETF数(互斥) | 30~500 | 检查因子间相似度时抽样多少只ETF。越多去重越准确 |
| GPU占用率上限 | 0.3~0.9 | GPU利用率上限。0.8=充分利用，0.5=留一半给其他任务。设太高可能导致电脑卡顿 |
| 内存上限(GB) | 4~16 | Python进程最大可用内存。8GB适合16G内存机器，4GB适合8G内存机器 |
| 最大并发数 | 1~4 | 同时运行的因子计算任务数。1=安全稳定，2~4=更快但更吃资源 |

**最大ETF数(IC)详解：**
- 决定用多少只ETF来计算"因子能否预测收益"
- 数值小（如50）：IC值波动大、偏乐观（小样本偏差），但计算快
- 数值大（如500）：IC值更接近真实水平，但计算慢
- 实测：50只ETF时IC约-0.28，200只时约-0.27，全量1036只时约-0.05

**最大ETF数(互斥)详解：**
- 决定用多少只ETF来计算"两个因子是否在说同一件事"（互相关IC）
- 数值小（如30）：可能漏判相似因子（把实质相同的因子当成不同的保留下来），导致因子冗余
- 数值大（如200）：能更准确识别相似因子，去重效果更好，但计算耗时增加
- 互斥IC计算量 = 因子对数 × ETF数，因子多时影响显著
- 建议：设为最大ETF数(IC)的一半左右即可，因为去重精度要求略低于IC评估

**两个ETF数参数的区别：**
- **最大ETF数(IC)** → 评估"因子质量"：这个因子能不能预测收益？
- **最大ETF数(互斥)** → 评估"因子独立性"：这两个因子是不是在说同一件事？
- 前者影响IC值大小和有效性判断，后者影响去重效果和最终保留多少因子

**实测数据（当前ETF池共1036只，volatility_20d因子）：**
| 最大ETF数(IC) | 单因子耗时 | IC结果 | 适用场景 |
|------|-----------|--------|----------|
| 50只 | ~0.8s | 偏高（小样本偏差） | 快速验证 |
| 100只 | ~0.8s | 较准确 | 日常使用（推荐） |
| 200只 | ~2.5s | 准确 | 精确评估 |
| 500只 | ~8.7s | 很准确 | 深度研究 |
| 全量1036只 | ~17s | 最准确 | 论文级研究 |

**关于计算速度：**
当前"生成因子"是对15个预置表达式逐一计算IC/RankIC，使用pandas+scipy纯CPU运算，因此速度较快。"因子挖掘"使用UCB1引导搜索+遗传编程自动发现新因子，同样基于CPU运算，支持表达式预筛选加速。
"""

_MINING_HINTS = """
**因子挖掘参数说明：**

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| 搜索步数(N-steps) | 512~8192 | UCB1搜索总步数。512=快速验证，2048=标准搜索，4096=深度搜索，8192=穷举搜索 |
| 批大小(batch_size) | 64~512 | 每步采样的因子表达式数量。越大搜索越稳定但越耗时 |
| 最大搜索时长(分钟) | 5~480 | 搜索超时保护。到达时限后自动停止并保存已发现的有效因子 |
| 最大因子数 | 10~100 | 搜索到多少个有效因子后自动停止 |

**搜索策略说明：**
- **UCB1搜索**：使用UCB1多臂老虎机算法智能探索因子表达式空间。自动平衡探索（尝试新算子组合）与利用（聚焦高成功率算子），同时利用n-gram条件概率实现上下文感知的算子选择，比随机搜索效率更高
- **遗传搜索**：模拟自然选择，对已有有效因子进行语法树交叉和变异产生新因子。若因子池为空（冷启动），会先用随机生成器创建初始种群，再进入遗传进化阶段
- **UCB1+遗传组合**：先用UCB1搜索发现初始因子，再用遗传搜索在已有因子基础上优化
- **因子组合搜索**：不搜索全新表达式，而是将因子池中已有的有效因子进行二元组合（相乘、相加、rank×sign等），寻找互补性更强的组合因子。搜索空间小、效率极高，适合因子池已有一定积累后使用
- **模因搜索**：在遗传搜索基础上增加局部搜索，对每个候选表达式的窗口参数和delta参数做穷举微调。很多因子"差一点就有效"（如窗口从20改为10），模因搜索能快速找到这些近优解

**算法改进（v2.0）：**
1. **UCB1置信上界**：替代硬编码的30%/70%探索/利用比例，根据统计置信度自动调节
2. **n-gram条件概率**：跟踪 P(算子有效 | 前一个算子)，实现上下文感知选择
3. **表达式预筛选**：快速评估1-2只ETF，跳过全NaN/常数表达式，节省约80%无效计算
4. **语法树交叉**：在最浅层二元运算符处拆分表达式，交换子表达式，比简单拼接更科学
5. **高价值模板**：内置10个量化研究经典因子模板（如标准化动量、截面排名×方向等）
6. **更深嵌套**：表达式深度从0-2层扩展到0-3层，可发现更复杂的因子结构

**资源消耗预估：**
| N-steps | batch_size | 预计耗时 | 适用场景 |
|---------|-----------|---------|----------|
| 512 | 64 | ~5分钟 | 快速验证 |
| 2048 | 64 | ~20分钟 | 标准搜索 |
| 4096 | 128 | ~60分钟 | 深度搜索 |
| 4096 | 512 | ~2小时 | 深度大batch搜索 |
| 8192 | 256 | ~2小时+ | 穷举搜索 |
"""

_FACTOR_LOGIC = {
    "momentum_20d": {
        "logic": "动量效应：过去20日涨幅较大的ETF，短期内倾向于延续上涨趋势。这是金融市场最经典的异象之一，源于投资者的羊群效应和正反馈交易。",
        "market": "趋势行情（牛市或明确上升通道），在震荡市中容易失效产生假信号。",
        "etf_type": "行业ETF、主题ETF（趋势延续性强），宽基ETF次之，债券ETF效果较弱。",
    },
    "momentum_5d": {
        "logic": "短期动量：捕捉5日内的价格趋势延续。相比20日动量更敏感，能更快捕捉趋势变化，但也更容易受噪音干扰。",
        "market": "短期趋势行情，适合波段操作。震荡市中频繁反转会导致亏损。",
        "etf_type": "高波动行业ETF（半导体、新能源等），低波动ETF信号较弱。",
    },
    "volatility_20d": {
        "logic": "波动率因子：20日历史波动率衡量ETF的风险水平。低波动率往往预示未来收益更稳定，高波动率可能意味着风险积聚或机会来临。",
        "market": "低波动异象在震荡市和弱市中有效（低波动ETF抗跌），高波动策略在牛市中更有效。",
        "etf_type": "宽基ETF（波动率稳定），行业ETF需结合趋势判断。",
    },
    "volatility_ratio": {
        "logic": "波动率比值：短期波动率/长期波动率。比值>1说明短期波动加剧，市场可能进入变盘期；比值<1说明短期趋于平静，可能酝酿突破。",
        "market": "变盘预警信号。比值从低位回升时，往往预示行情即将选择方向。",
        "etf_type": "所有ETF通用，特别适合波动率变化敏感的行业ETF。",
    },
    "volume_ratio_5d": {
        "logic": "量比因子：5日均量/20日均量。放量（比值>1）通常伴随趋势加速，缩量（比值<1）可能意味着趋势衰竭或蓄势。",
        "market": "放量确认趋势（牛市有效），缩量警惕反转。量价背离时信号最可靠。",
        "etf_type": "流动性好的宽基ETF和热门行业ETF，小规模ETF成交量信号不可靠。",
    },
    "turnover_momentum": {
        "logic": "价量相关性：衡量价格变化与成交量的相关程度。正相关说明量价齐升（健康上涨），负相关说明放量下跌（资金出逃）。",
        "market": "正相关在牛市中确认趋势，负相关在熊市中预警风险。震荡市中信号不明确。",
        "etf_type": "高流动性ETF（沪深300、中证500等），低流动性ETF量价关系失真。",
    },
    "high_low_range_20d": {
        "logic": "振幅因子：20日内最高价与最低价的差距。振幅收窄意味着市场犹豫，可能即将突破；振幅扩大意味着趋势加速。",
        "market": "振幅收窄后突破方向判断需结合其他因子。布林带收窄策略的变体。",
        "etf_type": "所有ETF通用，行业ETF振幅变化更明显。",
    },
    "close_to_high_ratio": {
        "logic": "区间位置因子：收盘价在20日高低区间的相对位置。接近1说明价格在区间顶部（强势），接近0说明在底部（弱势）。",
        "market": "趋势行情中，强势因子持续有效（强者恒强）；震荡行情中，接近顶部可能是反转信号。",
        "etf_type": "行业ETF和主题ETF（趋势延续性好），债券ETF区间位置变化缓慢。",
    },
    "mean_reversion_5d": {
        "logic": "均值回归因子：5日跌幅越大，因子值越大，押注短期反弹。逻辑是价格偏离均值后倾向于回归，是动量因子的对立面。",
        "market": "震荡市中非常有效（跌了会涨），趋势行情中会逆势亏损（越跌越买越亏）。",
        "etf_type": "宽基ETF（均值回归特性明显），行业ETF需谨慎（可能趋势性下跌）。",
    },
    "overnight_gap": {
        "logic": "隔夜跳空因子：今日开盘价相对昨日收盘价的跳空幅度。正跳空说明隔夜利好，负跳空说明隔夜利空。跳空方向往往预示当日走势。",
        "market": "消息驱动行情中有效（政策发布、外盘影响），平淡行情中信号微弱。",
        "etf_type": "受外盘影响大的ETF（港股通、QDII等），A股行业ETF隔夜跳空较小。",
    },
    "etf_premium_rate": {
        "logic": "折溢价率因子：ETF价格相对净值的偏离程度。溢价（>0）说明市场情绪偏乐观，折价（<0）说明偏悲观。极端折溢价会回归。",
        "market": "市场恐慌时大幅折价是买入机会，狂热时大幅溢价是卖出信号。日常波动中信号较弱。",
        "etf_type": "所有ETF专属因子！跨境ETF（QDII）折溢价最大最有效，A股ETF折溢价较小。",
    },
    "etf_tracking_error_20d": {
        "logic": "跟踪误差因子：ETF相对基准的偏离程度。跟踪误差增大说明ETF与基准走势分化，可能是套利机会或管理问题。",
        "market": "跟踪误差突然增大时值得关注，可能预示ETF成分股调整或市场异常。",
        "etf_type": "指数增强ETF、Smart Beta ETF（跟踪误差天然较大），纯被动ETF跟踪误差应很小。",
    },
    "etf_iopv_deviation": {
        "logic": "IOPV偏离度：ETF实时价格与IOPV（参考净值）的偏离。IOPV更准确反映ETF真实价值，偏离意味着交易价格失真。",
        "market": "盘中交易时有效，可用于日内T+0套利。开盘和收盘时段偏离最大。",
        "etf_type": "成分股停牌较多的ETF、跨境ETF，IOPV偏离更频繁。",
    },
    "etf_premium_momentum": {
        "logic": "折溢价动量：折溢价率5日变化趋势。溢价持续扩大说明市场情绪升温，折价持续扩大说明情绪降温。是情绪动量的代理指标。",
        "market": "情绪拐点判断：溢价由升转降可能是短期顶部，折价由升转降可能是短期底部。",
        "etf_type": "情绪敏感型ETF（科技、医药等），债券ETF折溢价变化太小。",
    },
    "etf_premium_volatility": {
        "logic": "折溢价波动率：折溢价率的20日波动程度。波动率上升说明市场对ETF定价分歧加大，可能预示变盘；波动率下降说明定价趋于一致。",
        "market": "折溢价波动率极低后回升，往往是行情启动的前兆。波动率极高后回落，可能是趋势衰竭。",
        "etf_type": "跨境ETF、商品ETF（折溢价波动大），A股宽基ETF波动较小。",
    },
}

_CATEGORY_LABELS = {
    "momentum": "动量",
    "volatility": "波动率",
    "volume": "成交量",
    "range": "振幅",
    "mean_reversion": "均值回归",
    "microstructure": "微观结构",
    "etf_specific": "ETF专属",
    "custom": "🔍挖掘",
    "mined": "🔍挖掘",
    "preset": "📐预置",
}


def create_factor_page(config: ETFQuantConfig) -> None:
    ui.label("🧬 因子管理").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = FactorService(config.alpha, config.data, config.ml)

    with ui.tabs().classes("full-width") as tabs:
        ui.tab("pool", label="因子池")
        ui.tab("generate", label="因子评估")
        ui.tab("mining", label="因子挖掘")
        ui.tab("train", label="筛选与训练")
        ui.tab("schedule", label="定时任务")

    with ui.tab_panels(tabs, value="pool").classes("full-width"):
        with ui.tab_panel("pool"):
            with ui.row().classes("q-mb-md items-center"):
                ui.button("🔄 刷新列表", on_click=lambda: _refresh_pool()).props("flat").style("color: #58a6ff").on("mouseover", lambda: ui.tooltip("重新从数据库读取因子列表，生成新因子后需点击此处更新"))
                ui.button("💾 导出Parquet", on_click=lambda: _export()).props("flat").style("color: #3fb950").on("mouseover", lambda: ui.tooltip("导出因子库为Parquet文件，可用于Jupyter分析或分享给团队"))
                ui.button("🗑 清空无效", on_click=lambda: _clear_invalid()).props("flat").style("color: #f85149").on("mouseover", lambda: ui.tooltip("删除有效性为❌的因子，不可恢复！"))
                ui.button("🔄 重新校验", on_click=lambda: _revalidate()).props("flat").style("color: #58a6ff").on("mouseover", lambda: ui.tooltip("按当前IC/ICIR阈值重新判定所有因子的有效性"))
                ui.button("🗑 删除选中", on_click=lambda: _delete_selected()).props("flat").style("color: #f85149").on("mouseover", lambda: ui.tooltip("删除勾选的因子，不可恢复！"))
                ui.button("🧹 清空同质", on_click=lambda: _find_redundant()).props("flat").style("color: #d29922").on("mouseover", lambda: ui.tooltip("找出因子值高度相关(>0.95)的冗余因子，每组保留IC最高的一个"))
                ui.space()
                pool_count_label = ui.label("").classes("text-caption").style("color: #8b949e")

            factor_table = ui.table(
                columns=[
                    {"name": "name", "label": "因子名", "field": "name", "sortable": True, "align": "left", "width": "120px"},
                    {"name": "ic", "label": "IC", "field": "ic", "sortable": True, "align": "right", "width": "65px"},
                    {"name": "rank_ic", "label": "RankIC", "field": "rank_ic", "sortable": True, "align": "right", "width": "65px"},
                    {"name": "icir", "label": "ICIR", "field": "icir", "sortable": True, "align": "right", "width": "70px"},
                    {"name": "is_valid", "label": "有效性", "field": "is_valid", "align": "center", "width": "60px"},
                    {"name": "category_label", "label": "分类", "field": "category_label", "sortable": True, "align": "center", "width": "80px"},
                    {"name": "updated_at", "label": "更新时间", "field": "updated_at", "sortable": True, "align": "center", "width": "140px"},
                ],
                rows=[],
                row_key="name",
                selection="multiple",
                pagination={"rowsPerPage": 5, "rowsPerPageOptions": [5, 10, 20, 50]},
            ).classes("full-width").props('resizable-columns separator="cell"')
            factor_table.on("rowClick", lambda e: _on_row_click(e))

            with ui.card().classes("full-width q-mt-md").style("border-radius: 12px !important; border: 1px solid #30363d !important;"):
                detail_title = ui.label("📋 点击因子查看详情").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
                detail_content = ui.label("").classes("text-body2").style("color: #8b949e; line-height: 1.8; white-space: pre-line;")

            def _on_row_click(e):
                if not e.args:
                    return
                args = e.args
                if isinstance(args, list) and len(args) >= 2:
                    row = args[1]
                elif isinstance(args, dict):
                    row = args
                else:
                    return
                name = row.get("name", "") if isinstance(row, dict) else str(row)
                if not name:
                    return
                factor = svc.get_factor(name)
                if not factor:
                    return
                cat = factor.get("category", "")
                cat_label = _CATEGORY_LABELS.get(cat, cat)
                valid_text = "✅ 有效" if factor.get("is_valid") else "❌ 无效"
                lines = [
                    f"📌 因子名称：{factor['name']}",
                    f"📝 表达式：{factor.get('expression', '')}",
                    f"📊 IC = {factor.get('ic', 0):.4f}  |  RankIC = {factor.get('rank_ic', 0):.4f}  |  ICIR = {factor.get('icir', 0):.4f}  |  {valid_text}",
                    f"🏷 分类：{cat_label}  |  更新时间：{factor.get('updated_at', '-')}",
                ]
                logic_info = _FACTOR_LOGIC.get(factor["name"])
                if logic_info:
                    lines += [
                        "",
                        "🧠 因子逻辑：" + logic_info["logic"],
                        "",
                        "📈 适用行情：" + logic_info["market"],
                        "",
                        "🎯 适用ETF：" + logic_info["etf_type"],
                    ]
                elif factor.get("description"):
                    lines += ["", f"📝 说明：{factor['description']}"]
                detail_title.text = f"📋 {factor['name']}"
                detail_content.text = "\n".join(lines)

            def _refresh_pool():
                rows = svc.list_factors()
                for r in rows:
                    r["ic"] = f"{r.get('ic') or 0:.4f}"
                    r["rank_ic"] = f"{r.get('rank_ic') or 0:.4f}"
                    r["icir"] = f"{r.get('icir') or 0:.4f}"
                    r["is_valid"] = "✅" if r.get("is_valid") else "❌"
                    r["category_label"] = _CATEGORY_LABELS.get(r.get("category", ""), r.get("category", ""))
                    ts = r.get("updated_at", "")
                    if "T" in ts:
                        ts = ts.replace("T", " ")[:19]
                    r["updated_at"] = ts
                factor_table.rows = rows
                valid_count = sum(1 for r in rows if r["is_valid"] == "✅")
                pool_count_label.text = f"共 {len(rows)} 个因子，{valid_count} 个有效"

            def _export():
                path = svc.export_factors()
                ui.notify(f"已导出: {path}", type="positive")

            def _clear_invalid():
                count = svc.clear_invalid()
                ui.notify(f"已清除 {count} 个无效因子", type="positive")
                _refresh_pool()

            def _delete_selected():
                selected = factor_table.selected
                if not selected:
                    ui.notify("请先勾选要删除的因子", type="warning")
                    return
                names = [r["name"] if isinstance(r, dict) else r for r in selected]
                count = 0
                for name in names:
                    if svc.delete_factor(name):
                        count += 1
                ui.notify(f"已删除 {count} 个因子", type="positive")
                factor_table.selected = []
                _refresh_pool()

            def _revalidate():
                result = svc.revalidate_all(
                    ic_threshold=ic_input.value,
                    rank_ic_threshold=ric_input.value,
                    icir_threshold=icir_input.value,
                )
                ui.notify(f"校验完成: {result['total']}个因子中{result['updated']}个有效性变更", type="positive")
                _refresh_pool()

            _redundant_result = {"groups": None, "computing": False}

            def _find_redundant():
                if _redundant_result["computing"]:
                    ui.notify("正在计算中，请稍候...", type="warning")
                    return
                _redundant_result["computing"] = True
                _redundant_result["groups"] = None
                ui.notify("正在计算因子间相关性，请稍候...", type="info")

                def _compute():
                    try:
                        groups = svc.find_redundant_factors(corr_threshold=0.95)
                        _redundant_result["groups"] = groups
                    except Exception as e:
                        _redundant_result["groups"] = {"error": str(e)}
                    finally:
                        _redundant_result["computing"] = False

                import asyncio
                asyncio.get_event_loop().run_in_executor(None, _compute)

            def _poll_redundant():
                if _redundant_result["computing"]:
                    return
                groups = _redundant_result["groups"]
                if groups is None:
                    return
                _redundant_result["groups"] = None
                if isinstance(groups, dict) and "error" in groups:
                    ui.notify(f"计算失败: {groups['error']}", type="negative")
                    return
                if not groups:
                    ui.notify("未发现同质因子，因子池多样性良好！", type="positive")
                    return
                total_remove = sum(g["remove_count"] for g in groups)
                with ui.dialog() as dialog, ui.card().style("min-width: 500px; max-height: 70vh;"):
                    ui.label(f"发现 {len(groups)} 组同质因子，共 {total_remove} 个冗余").classes("text-h6 q-mb-sm").style("color: #d29922")
                    ui.label("每组保留IC最高的因子，删除其余冗余因子").classes("text-body2 q-mb-md").style("color: #8b949e")
                    with ui.scroll_area().style("max-height: 400px;"):
                        for i, g in enumerate(groups):
                            with ui.expansion(f"第{i+1}组: 保留 {g['keep']} (IC={g['keep_ic']:.4f}), 删除 {g['remove_count']}个", icon="group_work").classes("full-width q-mb-xs").style("border: 1px solid #30363d; border-radius: 4px;"):
                                for name in g["remove"]:
                                    ui.label(f"  🗑 {name}").style("color: #f85149; font-size: 12px;")
                                ui.label(f"  组内最大相关系数: {g['max_corr']:.4f}").style("color: #8b949e; font-size: 12px;")
                    with ui.row().classes("q-mt-md"):
                        ui.button("取消", on_click=dialog.close).props("flat")
                        ui.button(f"确认删除 {total_remove} 个冗余因子", on_click=lambda: [_do_remove_redundant(groups), dialog.close()], color="negative")
                dialog.open()

            ui.timer(1.0, _poll_redundant)

            def _do_remove_redundant(groups):
                removed = svc.remove_redundant_factors(groups)
                ui.notify(f"已删除 {removed} 个冗余因子", type="positive")
                _refresh_pool()

            _refresh_pool()

        with ui.tab_panel("generate"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("预置因子评估").classes("text-h6 q-mb-sm").style("color: #58a6ff")
                ui.label("对15个预置因子表达式重新计算IC/RankIC（非搜索新因子），参数变化会影响评估结果").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row().classes("full-width"):
                    ic_input = ui.number(label="IC阈值", value=config.alpha.ic_threshold, format="%.4f", step=0.01, min=0.0, max=1.0).classes("q-mr-md").style("min-width: 160px")
                    ric_input = ui.number(label="RankIC阈值", value=config.alpha.rank_ic_threshold, format="%.4f", step=0.01, min=0.0, max=1.0).classes("q-mr-md").style("min-width: 160px")
                    icir_input = ui.number(label="ICIR阈值", value=config.alpha.icir_threshold, format="%.2f", step=0.1, min=0.0, max=10.0).classes("q-mr-md").style("min-width: 160px")
                with ui.row().classes("full-width q-mt-md"):
                    period_input = ui.number(label="预测周期", value=config.alpha.target_period, min=1, max=60, step=1).classes("q-mr-md").style("min-width: 160px")
                    max_etf_ic_input = ui.number(label="最大ETF数(IC)", value=config.alpha.max_etf_for_ic, min=5, max=2000, step=5).classes("q-mr-md").style("min-width: 160px")
                    max_etf_mutual_input = ui.number(label="最大ETF数(互斥)", value=config.alpha.max_etf_for_mutual_ic, min=5, max=2000, step=5).classes("q-mr-md").style("min-width: 160px")
                with ui.row().classes("full-width q-mt-md"):
                    gpu_input = ui.number(label="GPU占用率上限", value=config.alpha.resources.gpu_utilization_limit, format="%.1f", min=0.1, max=1.0, step=0.1).classes("q-mr-md").style("min-width: 180px")
                    mem_input = ui.number(label="内存上限(GB)", value=config.alpha.resources.memory_limit_gb, format="%.1f", min=1, max=64, step=1.0).classes("q-mr-md").style("min-width: 180px")
                    conc_input = ui.number(label="最大并发数", value=config.alpha.resources.max_concurrent_tasks, min=1, max=8, step=1).classes("q-mr-md").style("min-width: 180px")
                ui.button("🧬 生成因子", on_click=lambda: _generate(), color="primary").classes("q-mt-md")
                gen_progress = ui.linear_progress(value=0.0, size="6px").classes("q-mt-sm").style("display: none;")
                gen_status = ui.label("").classes("text-body2 q-mt-xs").style("color: #8b949e")

            with ui.expansion("📖 参数说明与建议范围", icon="help").classes("full-width q-mb-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                ui.markdown(_PARAM_HINTS).style("color: #c9d1d9; font-size: 13px;")

            with ui.card().classes("full-width q-mb-md"):
                ui.label("自定义因子表达式").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.markdown(
                    "在下方输入因子表达式，系统会计算该表达式对所有ETF的 **IC** 和 **RankIC**。\n\n"
                    "- **等号左边**：因子值 = 表达式计算结果（系统自动命名）\n"
                    "- **评估按钮**：计算该表达式的 IC/RankIC，判断其预测能力\n"
                    "- **IC > 0.03** 表示因子有效，**> 0.05** 表示优秀\n\n"
                    "示例：`premium_rate() * ts_return(close, 5)` 表示溢价率乘以5日收益率"
                ).style("color: #8b949e; font-size: 13px; line-height: 1.6;")
                expr_input = ui.input(label="因子表达式", placeholder="如: premium_rate() * ts_return(close, 5)").classes("full-width q-mb-sm").props("outlined")
                with ui.row().classes("items-center"):
                    ui.button("📊 评估因子", on_click=lambda: _eval_custom(), color="primary").props("dense")
                eval_result = ui.label("").classes("text-body2 q-mt-sm").style("color: #8b949e")

            with ui.expansion("📖 算子参考与因子表达式指南", icon="help").classes("full-width q-mb-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                ui.markdown(
                    "在上方自定义因子表达式中，你可以使用以下算子来组合出新的因子。\n\n"
                    "**💡 写因子的思路：**\n"
                    "1. 选一个你感兴趣的信号（如折溢价率、动量、波动率）\n"
                    "2. 用时序算子加工（如求均值、排名、差值）\n"
                    "3. 多个信号可以相乘、相加、相减\n"
                    "4. 写好后点【评估因子】按钮，看 IC 是否 > 0.03\n\n"
                    "**可用的行情变量：** `close`, `open`, `high`, `low`, `volume`, `amount`, `pct_chg`\n\n"
                    "**可用的ETF变量：** `nav`（净值）, `iopv`（参考净值）, `index_close`（指数收盘价）\n\n"
                    "**数学函数：** `abs()`, `log()`, `sign()`, `power(x, n)`, `max()`, `min()`"
                ).style("color: #c9d1d9; font-size: 13px; line-height: 1.8;")
                operators = svc.get_operators()
                ui.table(
                    columns=[
                        {"name": "op", "label": "算子", "field": "op", "align": "left", "width": "180px"},
                        {"name": "name", "label": "名称", "field": "name", "align": "left", "width": "100px"},
                        {"name": "desc", "label": "说明", "field": "desc", "align": "left"},
                        {"name": "example", "label": "示例", "field": "example", "align": "left", "width": "240px"},
                        {"name": "tip", "label": "提示", "field": "tip", "align": "left", "width": "200px"},
                    ],
                    rows=operators,
                ).classes("full-width")

            async def _generate():
                gen_status.text = "⏳ 正在准备..."
                gen_status.style("color: #58a6ff")
                gen_progress.value = 0.0
                gen_progress.style("display: block;")
                shared = {"pct": 0, "msg": "准备中...", "done": False, "result": None, "error": None}

                def on_progress(current: int, total: int, name: str) -> None:
                    pct = int(current / total * 100) if total > 0 else 0
                    shared["pct"] = pct
                    shared["msg"] = f"正在评估 [{current}/{total}] {name}"

                def _run():
                    try:
                        r = svc.generate_preset_factors(
                            ic_threshold=ic_input.value,
                            rank_ic_threshold=ric_input.value,
                            icir_threshold=icir_input.value,
                            target_period=period_input.value,
                            max_etf_for_ic=max_etf_ic_input.value,
                            max_etf_for_mutual_ic=max_etf_mutual_input.value,
                            progress_callback=on_progress,
                        )
                        shared["result"] = r
                    except Exception as e:
                        shared["error"] = str(e)
                    finally:
                        shared["done"] = True

                timer = ui.timer(0.5, lambda: _update_gen_progress(shared))

                def _update_gen_progress(state: dict) -> None:
                    if state["done"]:
                        timer.active = False
                        gen_progress.value = 1.0
                        if state["error"]:
                            gen_status.text = f"❌ 生成失败: {state['error']}"
                            gen_status.style("color: #f85149")
                            gen_progress.style("display: none;")
                        else:
                            r = state["result"]
                            gen_status.text = f"✅ 完成: {r['valid']} 个有效 / {r['total']} 个总因子"
                            gen_status.style("color: #3fb950")
                            _refresh_pool()
                    else:
                        gen_progress.value = state["pct"] / 100.0
                        gen_status.text = f"⏳ {state['msg']} {state['pct']}%"

                asyncio.get_event_loop().run_in_executor(None, _run)

            async def _eval_custom():
                expr = expr_input.value.strip()
                if not expr:
                    ui.notify("请输入因子表达式", type="warning")
                    return
                eval_result.text = "⏳ 正在评估..."
                eval_result.style("color: #58a6ff")
                try:
                    result = await asyncio.get_event_loop().run_in_executor(None, lambda: svc.evaluate_expression(expr))
                    ic = result.get("ic") or 0
                    ric = result.get("rank_ic") or 0
                    icir = result.get("icir") or 0
                    ic_p = result.get("ic_p") or 1
                    sig = "显著" if ic_p <= 0.05 else "不显著"
                    if abs(ic) > 0.05 and abs(icir) > 0.5 and ic_p <= 0.05:
                        verdict = "🌟 优秀因子"
                    elif abs(ic) > 0.03 and abs(icir) > 0.3 and ic_p <= 0.05:
                        verdict = "✅ 有效因子"
                    elif abs(ic) > 0.02 and ic_p <= 0.10:
                        verdict = "⚠️ 弱因子"
                    else:
                        verdict = "❌ 无效因子"
                    eval_result.text = f"IC={ic:.4f} | RankIC={ric:.4f} | ICIR={icir:.2f} | p={ic_p:.3f}({sig}) | {verdict}"
                    eval_result.style("color: #3fb950" if verdict.startswith("✅") or verdict.startswith("🌟") else "#d29922" if verdict.startswith("⚠️") else "#f85149")
                except Exception as e:
                    eval_result.text = f"❌ 评估失败: {e}"
                    eval_result.style("color: #f85149")

        with ui.tab_panel("mining"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("因子挖掘").classes("text-h6 q-mb-sm").style("color: #58a6ff")
                ui.label("自动搜索新的有效因子表达式，发现后自动存入因子池").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row().classes("full-width"):
                    nsteps_input = ui.number(label="搜索步数(N-steps)", value=4096, min=64, max=8192, step=256).classes("q-mr-md").style("min-width: 200px")
                    batch_input = ui.number(label="批大小(batch_size)", value=512, min=16, max=1024, step=16).classes("q-mr-md").style("min-width: 200px")
                with ui.row().classes("full-width q-mt-md"):
                    timeout_input = ui.number(label="最大搜索时长(分钟)", value=480, min=1, max=480, step=5).classes("q-mr-md").style("min-width: 200px")
                    max_mining_input = ui.number(label="最大因子数", value=100, min=1, max=200, step=5).classes("q-mr-md").style("min-width: 200px")
                with ui.row().classes("full-width q-mt-md items-center"):
                    strategy_select = ui.select(label="搜索策略", options=["UCB1搜索", "遗传搜索", "UCB1+遗传组合", "因子组合搜索", "模因搜索"], value="UCB1+遗传组合").classes("q-mr-md").style("min-width: 200px")
                with ui.row().classes("q-mt-md items-center"):
                    mining_btn = ui.button("🚀 开始挖掘", on_click=lambda: _start_mining(), color="primary")
                    stop_mining_btn = ui.button("⏹ 停止挖掘", on_click=lambda: _stop_mining(), color="negative").style("display: none;")
                mining_progress = ui.linear_progress(value=0.0, size="6px").classes("q-mt-sm").style("display: none;")
                mining_status = ui.label("").classes("text-body2 q-mt-xs").style("color: #8b949e")
                mining_log = ui.column().classes("full-width q-mt-sm").style("max-height: 180px; overflow-y: auto; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px;")
                mining_log_lines: list[str] = []

            def _mining_log(msg: str) -> None:
                import time as _t
                ts = _t.strftime("%H:%M:%S")
                line = f"[{ts}] {msg}"
                mining_log_lines.append(line)
                if len(mining_log_lines) > 100:
                    mining_log_lines.pop(0)
                mining_log.clear()
                with mining_log:
                    for l in mining_log_lines[-30:]:
                        ui.label(l).classes("text-caption").style("color: #8b949e; font-family: monospace; line-height: 1.4;")

            with ui.expansion("📖 因子挖掘参数说明", icon="help").classes("full-width q-mb-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                ui.markdown(_MINING_HINTS).style("color: #c9d1d9; font-size: 13px;")

            def _set_mining_ui_running(running: bool) -> None:
                if running:
                    mining_btn.text = "⏳ 挖掘进行中..."
                    mining_btn.props("color=orange disable")
                    stop_mining_btn.style("display: inline-flex;")
                    mining_progress.style("display: block;")
                else:
                    mining_btn.text = "🚀 开始挖掘"
                    mining_btn.props("color=primary")
                    mining_btn.props(remove="disable")
                    stop_mining_btn.style("display: none;")

            def _stop_mining() -> None:
                result = svc.stop_mining()
                if result.get("status") == "cancelling":
                    mining_status.text = "⏹ 正在停止挖掘（等待当前评估完成）..."
                    mining_status.style("color: #d29922")
                    ui.notify("已发送停止信号，挖掘将在当前评估完成后停止", type="info")
                else:
                    ui.notify("当前没有正在执行的挖掘任务", type="warning")

            def _start_mining() -> None:
                status = svc.get_mining_status()
                if status.get("running"):
                    ui.notify("挖掘任务正在执行中，请等待完成", type="warning")
                    return
                _set_mining_ui_running(True)
                mining_status.text = "⏳ 正在准备..."
                mining_status.style("color: #58a6ff")
                mining_progress.value = 0.0

                def _run():
                    try:
                        svc.mine_factors(
                            n_steps=int(nsteps_input.value),
                            batch_size=int(batch_input.value),
                            timeout_minutes=timeout_input.value,
                            max_factors=int(max_mining_input.value),
                            strategy=strategy_select.value,
                            ic_threshold=ic_input.value,
                            rank_ic_threshold=ric_input.value,
                            icir_threshold=icir_input.value,
                            target_period=period_input.value,
                            max_etf_for_ic=max_etf_ic_input.value,
                            max_etf_for_mutual_ic=max_etf_mutual_input.value,
                        )
                    except Exception as e:
                        logger.error("挖掘异常: %s", e)

                asyncio.get_event_loop().run_in_executor(None, _run)

            _notified_mining_done = {"done": False}
            _last_logged_pct = {"pct": -1}

            def _poll_mining_status() -> None:
                status = svc.get_mining_status()
                if status.get("running"):
                    _notified_mining_done["done"] = False
                    _set_mining_ui_running(True)
                    pct = status.get("pct", 0)
                    msg = status.get("msg", "")
                    mining_progress.value = pct / 100.0
                    start = status.get("start_time", "")
                    elapsed_hint = f" (启动于 {start[:19]})" if start else ""
                    mining_status.text = f"⏳ {msg} {pct}%{elapsed_hint}"
                    mining_status.style("color: #58a6ff")
                    if pct // 10 != _last_logged_pct["pct"] // 10 and pct > 0:
                        _last_logged_pct["pct"] = pct
                        _mining_log(f"{msg} {pct}%")
                elif status.get("result"):
                    _set_mining_ui_running(False)
                    r = status["result"]
                    elapsed = r.get("elapsed_seconds", 0) or 0
                    n_valid = r.get("total_valid", 0) or 0
                    n_eval = r.get("total_evaluated", 0) or 0
                    best_ic = r.get("best_ic", 0) or 0
                    mining_progress.value = 1.0
                    if r.get("cancelled"):
                        mining_status.text = f"⏹ 已停止: 评估{n_eval}个, 发现{n_valid}个有效因子, 耗时{elapsed:.0f}s, 最佳IC={best_ic:.4f}"
                        mining_status.style("color: #d29922")
                    else:
                        mining_status.text = f"✅ 完成: 评估{n_eval}个, 发现{n_valid}个有效因子, 耗时{elapsed:.0f}s, 最佳IC={best_ic:.4f}"
                        mining_status.style("color: #3fb950")
                    if not _notified_mining_done["done"]:
                        _notified_mining_done["done"] = True
                        _mining_log(f"✅ 完成: 评估{n_eval}个, 发现{n_valid}个有效因子, 耗时{elapsed:.0f}s, 最佳IC={best_ic:.4f}")
                        ui.notify(f"🎉 因子挖掘完成！发现{n_valid}个有效因子，最佳IC={best_ic:.4f}", type="positive", timeout=10000)
                        _refresh_pool()
                elif status.get("error"):
                    _set_mining_ui_running(False)
                    mining_status.text = f"❌ 挖掘失败: {status['error']}"
                    mining_status.style("color: #f85149")
                    if not _notified_mining_done["done"]:
                        _notified_mining_done["done"] = True
                        _mining_log(f"❌ 挖掘失败: {status['error']}")
                        ui.notify(f"❌ 因子挖掘失败: {status['error']}", type="negative", timeout=10000)
                    mining_progress.style("display: none;")

            ui.timer(2.0, _poll_mining_status)

        with ui.tab_panel("train"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("第一步：因子筛选").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.label("三级漏斗筛选：IC筛选 → ICIR筛选 → 去相关筛选。筛选结果自动保存，下方训练模型时会自动使用筛选后的因子作为特征。因子池不受影响，被筛掉的因子不会被删除。").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row():
                    s_ic = ui.number(label="IC阈值", value=config.ml.factor_screen.ic_threshold, format="%.2f", step=0.01, min=0.0, max=1.0).classes("q-mr-md").style("min-width: 160px")
                    s_ric = ui.number(label="RankIC阈值", value=config.ml.factor_screen.rank_ic_threshold, format="%.2f", step=0.01, min=0.0, max=1.0).classes("q-mr-md").style("min-width: 160px")
                    s_icir = ui.number(label="ICIR阈值", value=config.ml.factor_screen.icir_threshold, format="%.2f", step=0.1, min=0.0, max=10.0).classes("q-mr-md").style("min-width: 160px")
                    s_mutual = ui.number(label="互斥IC阈值", value=config.ml.factor_screen.mutual_ic_threshold, format="%.2f", step=0.05, min=0.0, max=1.0).classes("q-mr-md").style("min-width: 160px")
                    s_max = ui.number(label="最大因子数", value=config.ml.factor_screen.max_factors, step=1, min=1, max=100).classes("q-mr-md").style("min-width: 160px")
                ui.button("🔍 执行筛选", on_click=lambda: _screen(), color="primary").classes("q-mt-md")

            with ui.expansion("📖 因子筛选参数说明", icon="help").classes("full-width q-mb-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                ui.markdown("""
**因子筛选三级漏斗说明：**

| 层级 | 参数 | 建议范围 | 实际可取值范围 | 说明 |
|------|------|----------|--------------|------|
| 第一级 | IC阈值 | 0.02~0.05 | 0.00~1.00 | 因子IC绝对值需超过此值。0.03=常规标准，0.05=严格标准 |
| 第二级 | ICIR阈值 | 0.3~1.0 | 0.00~10.00 | 因子ICIR绝对值需超过此值。0.5=稳定，1.0=非常稳定 |
| 第三级 | 互斥IC阈值 | 0.5~0.8 | 0.00~1.00 | 两因子截面相关系数超过此值则视为相似，只保留IC更高的 |
| 最终 | 最大因子数 | 10~30 | 1~100 | 最终保留的因子数量上限。太多会过拟合，太少信息不足 |

**互斥IC阈值详解：**
- **值越大**：去重越宽松，会保留更多相似因子，可能导致因子冗余，模型稳定性下降
- **值越小**：去重越严格，可能过度去冗余，丢失互补信息，因子多样性不足
- |corr| > 0.7 → 两个因子在说同一件事，只保留IC更高的
- |corr| < 0.5 → 两个因子提供不同维度的信息，都保留
- 设得太低(如0.3)会过度去冗余，可能丢失互补信息
- 设得太高(如0.9)会保留太多相似因子，导致模型不稳定
""").style("color: #c9d1d9; font-size: 13px; line-height: 1.6;")

            screen_result = ui.label("").classes("text-body1 q-mb-md").style("color: #8b949e")

            with ui.row().classes("items-center q-mb-sm"):
                ui.label("筛选结果").classes("text-subtitle2").style("color: #8b949e")
                ui.space()
                screen_select_hint = ui.label("").classes("text-caption").style("color: #58a6ff")

            screen_table = ui.table(
                columns=[
                    {"name": "name", "label": "因子名", "field": "name", "sortable": True, "align": "left", "width": "120px"},
                    {"name": "ic", "label": "IC", "field": "ic", "sortable": True, "align": "right", "width": "65px"},
                    {"name": "rank_ic", "label": "RankIC", "field": "rank_ic", "sortable": True, "align": "right", "width": "65px"},
                    {"name": "icir", "label": "ICIR", "field": "icir", "sortable": True, "align": "right", "width": "70px"},
                ],
                rows=[],
                row_key="name",
                selection="multiple",
                pagination={"rowsPerPage": 20, "rowsPerPageOptions": [10, 20, 50, 100]},
            ).classes("full-width").props('resizable-columns separator="cell"')
            screen_table.on("selection", lambda e: _on_screen_select(e))

            ui.separator().classes("q-my-lg")

            with ui.card().classes("full-width q-mb-md"):
                ui.label("第二步：模型训练").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.label("基于筛选后的因子训练 XGBoost 预测模型。勾选上方筛选结果中的因子后点击'使用选中因子训练'；若未勾选，则使用全部筛选结果。").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row().classes("full-width items-end"):
                    etf_count_input = ui.number(label="ETF数量", value=50, min=5, max=500).classes("q-mr-md").style("min-width: 160px")
                    predict_input = ui.number(label="预测天数", value=config.ml.predict_days, min=1, max=20).classes("q-mr-md").style("min-width: 160px")
                    ui.button("🤖 使用选中因子训练", on_click=lambda: _train_model(use_selected=True), color="primary").classes("q-mr-md")
                    ui.button("🤖 使用全部筛选结果训练", on_click=lambda: _train_model(use_selected=False), color="secondary").classes("q-mr-md")
                train_status = ui.label("").classes("text-body2 q-mt-sm").style("color: #8b949e")

            with ui.expansion("📖 模型训练参数说明", icon="help").classes("full-width q-mb-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                ui.markdown("""
**模型训练参数说明：**

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| ETF数量 | 20~200 | 参与训练的ETF数量。越多样本越丰富但耗时越长。50=快速验证，100=标准训练，200=深度训练 |
| 预测天数 | 1~20 | 模型预测未来N天的收益率。1=日频交易，5=周频交易（推荐），10=双周频，20=月频 |

**训练流程：**
1. 从因子筛选结果中读取入选因子表达式（若无筛选结果则使用内置硬编码特征）
2. 对每只ETF计算因子值作为特征，计算未来N天收益率作为标签
3. 按时间序列划分训练集(80%)和验证集(20%)
4. 使用XGBoost回归模型训练，StandardScaler标准化特征
5. 保存模型到本地，可在回测页加载使用

**特征来源：**
- **筛选因子**：若已执行因子筛选，自动使用筛选结果中的因子表达式计算特征值
- **硬编码特征**：若未筛选，使用内置技术指标（收益率、均线、波动率、折溢价等）

**预测天数选择建议：**
- 1天：日频交易，换手率高，佣金成本大，适合大资金
- 5天：**周频交易（推荐）**，换手率适中，佣金可控，适合中小资金
- 10~20天：低频交易，信号少但稳定，适合长期持有

**ETF数量选择建议：**
- <20只：样本太少，模型容易过拟合
- 50只：快速验证模型框架是否正常
- 100只：标准训练，平衡速度与质量
- 200只+：深度训练，耗时较长但模型更稳健
""").style("color: #c9d1d9; font-size: 13px; line-height: 1.6;")

            with ui.card().classes("full-width"):
                ui.label("已保存模型").classes("text-h6 q-mb-md").style("color: #58a6ff")
                model_table = ui.table(
                    columns=[
                        {"name": "name", "label": "模型名", "field": "name", "align": "left", "width": "260px"},
                        {"name": "path", "label": "存储地址", "field": "path", "align": "left", "width": "300px"},
                        {"name": "size_mb", "label": "大小(MB)", "field": "size_mb", "align": "center", "width": "80px"},
                    ],
                    rows=[],
                    row_key="name",
                    selection="multiple",
                ).classes("full-width").props('separator="cell"')
                model_table.on("rowClick", lambda e: _preview_model(e))
                with ui.row().classes("q-mt-sm"):
                    ui.button("🔄 刷新", on_click=lambda: _refresh_models()).props("flat").style("color: #58a6ff")
                    export_format = ui.select(label="导出格式", options=["json", "py", "md", "txt"], value="json").classes("q-mx-md").style("min-width: 100px")
                    ui.button("💾 导出选中", on_click=lambda: _export_model()).props("flat").style("color: #3fb950")

            def _screen():
                all_factors = svc.list_factors()
                if not all_factors:
                    screen_result.text = "因子库为空，请先生成因子"
                    screen_result.style("color: #f85149")
                    return
                screen_result.text = "⏳ 正在筛选因子..."
                screen_result.style("color: #58a6ff")

                async def _run():
                    import asyncio
                    loop = asyncio.get_event_loop()
                    selected = await loop.run_in_executor(None, _do_screen)
                    if selected is None:
                        screen_result.text = "❌ 筛选失败，请查看终端日志"
                        screen_result.style("color: #f85149")
                        return
                    selected_names = [s["name"] for s in selected]
                    svc.save_screen_result(
                        ic_threshold=s_ic.value,
                        icir_threshold=s_icir.value,
                        mutual_ic_threshold=s_mutual.value,
                        max_factors=int(s_max.value),
                        selected_names=selected_names,
                    )
                    selected_name_set = set(selected_names)
                    rows = []
                    for f in all_factors:
                        if f["name"] in selected_name_set:
                            rows.append({
                                "name": f["name"],
                                "ic": f"{f.get('ic', 0):.4f}",
                                "rank_ic": f"{f.get('rank_ic', 0):.4f}",
                                "icir": f"{f.get('icir', 0):.4f}",
                            })
                    screen_table.rows = rows
                    report = screen_instance.get_screening_report(all_factors, selected)
                    screen_result.text = (
                        f"✅ 筛选完成: IC筛选{report['ic_filtered']}/{len(all_factors)} → "
                        f"ICIR筛选{report['icir_filtered']}/{report['ic_filtered']} → "
                        f"去相关{len(selected)}/{report['icir_filtered']} → "
                        f"最终{len(selected)}个因子入选 "
                        f"（平均IC={report['selected_avg_ic']:.4f}, 平均ICIR={report['selected_avg_icir']:.2f}，可直接点击下方训练模型）"
                    )
                    screen_result.style("color: #3fb950")

                def _do_screen():
                    try:
                        return screen_instance.screen(all_factors)
                    except Exception as e:
                        logger.error("因子筛选异常: %s", e)
                        return None

                from etfquant.core.config import FactorScreenConfig
                from etfquant.data.bridge import DataBridge
                screen_config = FactorScreenConfig(
                    ic_threshold=s_ic.value,
                    rank_ic_threshold=s_ric.value,
                    icir_threshold=s_icir.value,
                    mutual_ic_threshold=s_mutual.value,
                    max_factors=int(s_max.value),
                )
                bridge = DataBridge(config.data)
                screen_instance = FactorScreener(screen_config, data_bridge=bridge)

                asyncio.get_event_loop().create_task(_run())

            def _on_screen_select(e):
                selected = screen_table.selected
                n = len(selected) if selected else 0
                if n > 0:
                    screen_select_hint.text = f"已选 {n} 个因子（点击'使用选中因子训练'仅用这些因子）"
                else:
                    screen_select_hint.text = "未选择，训练时将使用全部筛选结果"

            async def _train_model(use_selected: bool = False):
                selected_names = None
                if use_selected:
                    selected = screen_table.selected
                    if selected:
                        selected_names = [r["name"] if isinstance(r, dict) else r for r in selected]
                    else:
                        ui.notify("未勾选任何因子，将使用全部筛选结果训练", type="info")

                if not selected_names:
                    latest = svc.get_latest_screen_result()
                    if not latest or not latest.get("selected_names"):
                        train_status.text = "⚠️ 请先执行因子筛选（第一步），再训练模型。"
                        train_status.style("color: #d29922")
                        return
                    selected_names = latest["selected_names"]

                n_selected = len(selected_names)
                label = f"手动选择{n_selected}个因子" if use_selected and screen_table.selected else f"筛选结果{n_selected}个因子"
                train_status.text = f"⏳ 使用{label}训练中..."
                train_status.style("color: #58a6ff")
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: svc.train_model(etf_count=int(etf_count_input.value), predict_days=int(predict_input.value), factor_names=selected_names)
                    )
                    if result.get("success"):
                        feat_src = result.get("feature_source", "")
                        train_status.text = f"✅ 训练完成! 样本{result['train_samples']}+{result['val_samples']}, 特征{result['feature_count']}个({feat_src})"
                        train_status.style("color: #3fb950")
                        _refresh_models()
                    else:
                        train_status.text = f"❌ 训练失败: {result.get('error', '未知错误')}"
                        train_status.style("color: #f85149")
                except Exception as e:
                    train_status.text = f"❌ 异常: {e}"
                    train_status.style("color: #f85149")

            def _refresh_models():
                models = svc.list_saved_models()
                for m in models:
                    m["size_mb"] = f"{m['size_mb']:.2f}"
                model_table.rows = models

            def _export_model():
                selected = model_table.selected
                if not selected:
                    ui.notify("请先勾选要导出的模型", type="warning")
                    return
                fmt = export_format.value
                for row in selected:
                    name = row["name"] if isinstance(row, dict) else row
                    out_path = svc.export_model(name, fmt)
                    if out_path:
                        ui.notify(f"已导出: {out_path}", type="positive")
                    else:
                        ui.notify(f"导出失败: {name}", type="negative")

            def _preview_model(e):
                row = e.args
                name = row.get("name", "") if isinstance(row, dict) else ""
                if not name:
                    return
                detail = svc.get_model_detail(name)
                if not detail or detail.get("error"):
                    ui.notify(f"无法加载模型: {detail.get('error', '未知')}", type="negative")
                    return
                with ui.dialog() as dlg, ui.card().classes("q-pa-lg").style("min-width: 600px; max-width: 800px;"):
                    ui.label(f"📋 {name}").classes("text-h5 q-mb-md").style("color: #58a6ff")
                    feat_names = detail.get("feature_names", [])
                    feat_lines = "\n".join(f"  {i}. `{fn}`" for i, fn in enumerate(feat_names, 1)) if feat_names else "  （无）"
                    ui.markdown(f"""**模型类型**: `{detail.get('model_type', 'Unknown')}` ({detail.get('model_module', '')})

**标准化器**: `{detail.get('scaler_type', 'None')}`

**特征来源**: {detail.get('feature_source', '未知')}

**训练集**: {detail.get('train_samples', 0)} 样本 | {detail.get('train_period', '')}

**验证集**: {detail.get('val_samples', 0)} 样本 | {detail.get('val_period', '')}

**特征列表** ({detail.get('feature_count', 0)}个):
{feat_lines}

**文件**: {detail.get('file_path', '')} ({detail.get('file_size_kb', 0)} KB)
""").style("color: #c9d1d9; font-size: 13px; line-height: 1.8;")
                    ui.button("关闭", on_click=dlg.close).classes("q-mt-md")
                dlg.open()

            def _load_screen_history():
                latest = svc.get_latest_screen_result()
                if not latest or not latest.get("selected_names"):
                    return
                selected_names = set(latest["selected_names"])
                all_factors = svc.list_factors()
                rows = []
                for f in all_factors:
                    if f["name"] in selected_names:
                        rows.append({
                            "name": f["name"],
                            "ic": f"{f.get('ic', 0):.4f}",
                            "rank_ic": f"{f.get('rank_ic', 0):.4f}",
                            "icir": f"{f.get('icir', 0):.4f}",
                        })
                screen_table.rows = rows
                screen_time = latest.get("created_at", "")
                if "T" in screen_time:
                    screen_time = screen_time.replace("T", " ")[:19]
                screen_result.text = (
                    f"📋 上次筛选结果({len(selected_names)}个因子, "
                    f"筛选时间{screen_time}): "
                    f"IC阈值={latest.get('ic_threshold', 0):.4f}, "
                    f"ICIR阈值={latest.get('icir_threshold', 0):.2f}, "
                    f"互斥IC阈值={latest.get('mutual_ic_threshold', 0):.2f}"
                )
                screen_result.style("color: #58a6ff")

            _refresh_models()
            _load_screen_history()

        with ui.tab_panel("schedule"):
            with ui.card().classes("full-width"):
                ui.label("盘后自动因子挖掘").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.markdown(
                    "设置后，系统会在每个交易日的收盘后自动运行因子挖掘（先更新预置因子IC，再执行UCB1搜索挖掘新因子），并将结果记录到因子池中。每天在时间窗口内只执行一次。\n\n"
                    "- **启动**：保存设置并启动调度器\n"
                    "- **停止**：停止调度器，设置保留，下次可重新启动"
                ).style("color: #8b949e; font-size: 13px; line-height: 1.6;")
                with ui.row().classes("full-width q-mt-md"):
                    sched_enabled = ui.switch(text="启用定时任务", value=config.alpha.schedule.enabled).classes("q-mr-md")
                    start_input = ui.input(label="开始时间(HH:MM)", value=config.alpha.schedule.start_time).classes("q-mr-md").style("min-width: 160px")
                    end_input = ui.input(label="结束时间(HH:MM)", value=config.alpha.schedule.end_time).classes("q-mr-md").style("min-width: 160px")
                days_select = ui.select(
                    label="运行日",
                    options={"Monday": "周一", "Tuesday": "周二", "Wednesday": "周三", "Thursday": "周四", "Friday": "周五"},
                    multiple=True,
                    value=config.alpha.schedule.days,
                ).classes("full-width q-mt-md")
                with ui.row().classes("q-mt-md"):
                    ui.button("� 启动", on_click=lambda: _save_and_start(), color="primary")
                    ui.button("⏹ 停止", on_click=lambda: _stop_schedule(), color="negative")
                sched_status = ui.label("").classes("text-body2 q-mt-md").style("color: #8b949e")

            with ui.card().classes("full-width q-mt-md"):
                ui.label("挖掘历史记录").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.label("手动挖掘和定时挖掘的执行记录均在此展示").classes("text-body2 q-mb-md").style("color: #8b949e")
                mining_table = ui.table(
                    columns=[
                        {"name": "timestamp", "label": "执行时间", "field": "timestamp", "align": "left", "sortable": True, "width": "140px"},
                        {"name": "source", "label": "来源", "field": "source", "align": "center", "width": "80px"},
                        {"name": "strategy", "label": "策略", "field": "strategy", "align": "center", "width": "110px"},
                        {"name": "mined_valid", "label": "挖掘有效", "field": "mined_valid", "align": "center", "width": "80px"},
                        {"name": "elapsed", "label": "耗时(秒)", "field": "elapsed", "align": "center", "sortable": True, "width": "80px"},
                        {"name": "best_ic", "label": "最佳IC", "field": "best_ic", "align": "center", "sortable": True, "width": "70px"},
                    ],
                    rows=[],
                    row_key="timestamp",
                    pagination={"rowsPerPage": 10},
                ).classes("full-width").props('resizable-columns separator="cell"')
                ui.button("🔄 刷新记录", on_click=lambda: _refresh_mining_log()).props("flat").style("color: #58a6ff")

            def _refresh_sched_status():
                status = svc.get_schedule_status()
                days_cn = ", ".join(status.get("days_cn", status.get("days", [])))
                if status.get("running"):
                    sched_status.text = f"🟢 运行中 | 时间窗口: {status.get('schedule', '')} | 运行日: {days_cn} | 上次执行: {status.get('last_run', '无')}"
                    sched_status.style("color: #3fb950")
                elif status.get("enabled"):
                    sched_status.text = f"🔵 已启用 | 时间窗口: {status.get('schedule', '')} | 运行日: {days_cn}"
                    sched_status.style("color: #58a6ff")
                else:
                    sched_status.text = "⚪ 未启用"
                    sched_status.style("color: #8b949e")

            def _save_and_start():
                enabled = sched_enabled.value
                start = start_input.value.strip()
                end = end_input.value.strip()
                days = days_select.value
                if enabled and (not start or not end):
                    ui.notify("请填写开始和结束时间", type="warning")
                    return
                import re
                time_pattern = re.compile(r"^\d{1,2}:\d{2}$")
                if enabled:
                    if not time_pattern.match(start) or not time_pattern.match(end):
                        ui.notify("时间格式错误，请使用 HH:MM 格式（如 18:00）", type="warning")
                        return
                svc.update_schedule_config(enabled, start, end, days)
                if enabled:
                    sched_status.text = "✅ 调度器已启动"
                    sched_status.style("color: #3fb950")
                else:
                    sched_status.text = "✅ 设置已保存（未启用）"
                    sched_status.style("color: #58a6ff")
                ui.notify("设置已保存", type="positive")

            def _stop_schedule():
                svc.stop_schedule()
                sched_enabled.value = False
                sched_status.text = "调度器已停止（设置已保留）"
                sched_status.style("color: #f85149")

            def _refresh_mining_log():
                logs = svc.get_mining_log()
                rows = []
                for log in reversed(logs):
                    ts = log.get("timestamp", "")
                    if "T" in ts:
                        ts = ts.replace("T", " ")[:19]
                    rows.append({
                        "timestamp": ts,
                        "source": log.get("source", "定时任务" if log.get("preset_valid", 0) > 0 else "未知"),
                        "strategy": log.get("strategy", ""),
                        "mined_valid": log.get("total_valid", 0),
                        "elapsed": f"{log.get('elapsed_seconds', 0):.0f}",
                        "best_ic": f"{log.get('best_ic', 0):.4f}",
                    })
                mining_table.rows = rows

            _refresh_sched_status()
            _refresh_mining_log()
