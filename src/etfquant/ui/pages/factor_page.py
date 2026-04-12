from __future__ import annotations

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.factor import FactorService
from etfquant.ml.factor_screener import FactorScreener
from etfquant.ui.echart_theme import ECHART_GRID, ECHART_XAXIS, ECHART_YAXIS


def create_factor_page(config: ETFQuantConfig) -> None:
    ui.label("🧬 因子管理").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = FactorService(config.alpha, config.data, config.ml)
    screener = FactorScreener(config.ml.factor_screen)

    with ui.tabs().classes("full-width") as tabs:
        ui.tab("pool", label="因子池")
        ui.tab("generate", label="因子生成")
        ui.tab("screen", label="因子筛选")
        ui.tab("train", label="模型训练")
        ui.tab("schedule", label="定时任务")
        ui.tab("operators", label="算子参考")

    with ui.tab_panels(tabs, value="pool").classes("full-width"):
        with ui.tab_panel("pool"):
            with ui.row().classes("q-mb-md"):
                ui.button("🔄 刷新", on_click=lambda: _refresh_pool()).props("flat").style("color: #58a6ff")
                ui.button("💾 导出Parquet", on_click=lambda: _export()).props("flat").style("color: #3fb950")
                ui.button("🗑 清空无效", on_click=lambda: _clear_invalid()).props("flat").style("color: #f85149")

            factor_table = ui.table(
                columns=[
                    {"name": "name", "label": "因子名", "field": "name", "sortable": True, "align": "left"},
                    {"name": "expression", "label": "表达式", "field": "expression", "align": "left"},
                    {"name": "ic", "label": "IC", "field": "ic", "sortable": True},
                    {"name": "rank_ic", "label": "RankIC", "field": "rank_ic", "sortable": True},
                    {"name": "icir", "label": "ICIR", "field": "icir", "sortable": True},
                    {"name": "is_valid", "label": "有效", "field": "is_valid", "align": "center"},
                    {"name": "category", "label": "分类", "field": "category", "sortable": True, "align": "center"},
                    {"name": "updated_at", "label": "更新时间", "field": "updated_at", "align": "left"},
                ],
                rows=[],
                row_key="name",
                pagination={"rowsPerPage": 20},
            ).classes("full-width")

            ic_chart = ui.echart({}).classes("full-width q-mt-md")

            def _refresh_pool():
                rows = svc.list_factors()
                for r in rows:
                    r["ic"] = f"{r.get('ic', 0):.4f}"
                    r["rank_ic"] = f"{r.get('rank_ic', 0):.4f}"
                    r["icir"] = f"{r.get('icir', 0):.4f}"
                factor_table.rows = rows
                if rows:
                    names = [r["name"][:12] for r in rows[:20]]
                    ics = [r.get("ic", 0) for r in rows[:20]]
                    ic_chart._props['options'] = {
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "grid": ECHART_GRID,
                        "xAxis": {"type": "category", "data": names, **ECHART_XAXIS, "axisLabel": {"color": "#8b949e", "rotate": 30}},
                        "yAxis": {"type": "value", **ECHART_YAXIS},
                        "series": [{"type": "bar", "data": ics, "itemStyle": {"color": "#58a6ff"}}],
                    }
                    ic_chart.update()

            def _export():
                path = svc.export_factors()
                ui.notify(f"已导出: {path}", type="positive")

            def _clear_invalid():
                count = svc.clear_invalid()
                ui.notify(f"已清除 {count} 个无效因子", type="positive")
                _refresh_pool()

            _refresh_pool()

        with ui.tab_panel("generate"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("预置因子生成").classes("text-h6 q-mb-md").style("color: #58a6ff")
                with ui.row().classes("full-width"):
                    ic_input = ui.number(label="IC阈值", value=config.alpha.ic_threshold, format="%.4f").classes("q-mr-md")
                    ric_input = ui.number(label="RankIC阈值", value=config.alpha.rank_ic_threshold, format="%.4f").classes("q-mr-md")
                    icir_input = ui.number(label="ICIR阈值", value=config.alpha.icir_threshold, format="%.4f").classes("q-mr-md")
                    period_input = ui.number(label="预测周期", value=config.alpha.target_period).classes("q-mr-md")
                ui.button("🧬 生成因子", on_click=lambda: _generate(), color="primary").classes("q-mt-md")
                gen_progress = ui.linear_progress(value=0).classes("full-width q-mt-sm")
                gen_status = ui.label("").classes("text-body2 q-mt-sm").style("color: #8b949e")

            with ui.card().classes("full-width"):
                ui.label("自定义因子表达式").classes("text-h6 q-mb-md").style("color: #58a6ff")
                expr_input = ui.input(label="表达式", placeholder="如: premium_rate() * ts_return(close, 5)").classes("full-width q-mb-sm")
                ui.button("评估", on_click=lambda: _eval_custom()).props("flat").style("color: #58a6ff")
                eval_result = ui.label("").classes("text-body2 q-mt-sm").style("color: #8b949e")

            with ui.card().classes("full-width q-mt-md"):
                ui.label("资源限制").classes("text-h6 q-mb-md").style("color: #58a6ff")
                with ui.row():
                    ui.number(label="GPU占用率上限", value=config.alpha.resources.gpu_utilization_limit, format="%.1f").classes("q-mr-md")
                    ui.number(label="内存上限(GB)", value=config.alpha.resources.memory_limit_gb, format="%.1f").classes("q-mr-md")
                    ui.number(label="最大并发数", value=config.alpha.resources.max_concurrent_tasks).classes("q-mr-md")

            async def _generate():
                gen_status.text = "⏳ 正在生成因子..."
                gen_status.style("color: #58a6ff")
                gen_progress.value = 0.2
                try:
                    result = svc.generate_preset_factors()
                    gen_progress.value = 1.0
                    gen_status.text = f"✅ 完成: {result['valid']} 个有效 / {result['total']} 个总因子"
                    gen_status.style("color: #3fb950")
                    _refresh_pool()
                except Exception as e:
                    gen_progress.value = 0
                    gen_status.text = f"❌ 生成失败: {e}"
                    gen_status.style("color: #f85149")

            async def _eval_custom():
                expr = expr_input.value.strip()
                if not expr:
                    return
                result = svc.evaluate_expression(expr)
                eval_result.text = f"IC={result['ic']:.4f}, RankIC={result['rank_ic']:.4f}"

        with ui.tab_panel("screen"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("因子筛选配置").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.label("从因子池中筛选高IC、低相关的因子子集，供ML训练使用").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row():
                    s_ic = ui.number(label="IC阈值", value=config.ml.factor_screen.ic_threshold, format="%.4f").classes("q-mr-md")
                    s_icir = ui.number(label="ICIR阈值", value=config.ml.factor_screen.icir_threshold, format="%.4f").classes("q-mr-md")
                    s_mutual = ui.number(label="互斥IC阈值", value=config.ml.factor_screen.mutual_ic_threshold, format="%.4f").classes("q-mr-md")
                    s_max = ui.number(label="最大因子数", value=config.ml.factor_screen.max_factors).classes("q-mr-md")
                ui.button("🔍 执行筛选", on_click=lambda: _screen(), color="primary").classes("q-mt-md")

            screen_result = ui.label("").classes("text-body1 q-mb-md").style("color: #8b949e")

            screen_table = ui.table(
                columns=[
                    {"name": "name", "label": "因子名", "field": "name", "sortable": True, "align": "left"},
                    {"name": "expression", "label": "表达式", "field": "expression", "align": "left"},
                    {"name": "ic", "label": "IC", "field": "ic", "sortable": True},
                    {"name": "rank_ic", "label": "RankIC", "field": "rank_ic"},
                    {"name": "icir", "label": "ICIR", "field": "icir", "sortable": True},
                    {"name": "selected", "label": "入选", "field": "selected", "align": "center"},
                ],
                rows=[],
                row_key="name",
            ).classes("full-width")

            def _screen():
                all_factors = svc.list_factors()
                if not all_factors:
                    screen_result.text = "因子库为空，请先生成因子"
                    screen_result.style("color: #f85149")
                    return
                selected = screener.screen(all_factors)
                selected_names = {s["name"] for s in selected}
                rows = []
                for f in all_factors:
                    rows.append({
                        "name": f["name"],
                        "expression": f.get("expression", ""),
                        "ic": f"{f.get('ic', 0):.4f}",
                        "rank_ic": f"{f.get('rank_ic', 0):.4f}",
                        "icir": f"{f.get('icir', 0):.4f}",
                        "selected": "✅" if f["name"] in selected_names else "❌",
                    })
                screen_table.rows = rows
                screen_result.text = f"✅ 筛选完成: {len(selected)} 个因子入选 / {len(all_factors)} 个总因子"
                screen_result.style("color: #3fb950")

        with ui.tab_panel("train"):
            with ui.card().classes("full-width q-mb-md"):
                ui.label("ML 模型训练").classes("text-h6 q-mb-md").style("color: #58a6ff")
                ui.label("基于已筛选的因子，训练 XGBoost 预测模型，训练完成后可在回测页使用").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row().classes("full-width items-end"):
                    etf_count_input = ui.number(label="ETF数量", value=50, min=5, max=500).classes("q-mr-md")
                    predict_input = ui.number(label="预测天数", value=config.ml.predict_days, min=1, max=20).classes("q-mr-md")
                    ui.button("🤖 训练模型", on_click=lambda: _train_model(), color="primary").classes("q-mr-md")
                train_status = ui.label("").classes("text-body2 q-mt-sm").style("color: #8b949e")
                train_progress = ui.linear_progress(value=0).classes("full-width q-mt-sm")

            with ui.card().classes("full-width"):
                ui.label("已保存模型").classes("text-h6 q-mb-md").style("color: #58a6ff")
                model_table = ui.table(
                    columns=[
                        {"name": "name", "label": "模型名", "field": "name", "align": "left"},
                        {"name": "path", "label": "路径", "field": "path", "align": "left"},
                        {"name": "size_mb", "label": "大小(MB)", "field": "size_mb"},
                    ],
                    rows=[],
                ).classes("full-width")
                ui.button("🔄 刷新", on_click=lambda: _refresh_models()).props("flat").style("color: #58a6ff")

            async def _train_model():
                train_status.text = "⏳ 正在准备训练数据..."
                train_status.style("color: #58a6ff")
                train_progress.value = 0.2
                try:
                    result = svc.train_model()
                    train_progress.value = 1.0
                    if result.get("success"):
                        train_status.text = f"✅ 训练完成! 样本{result['train_samples']}+{result['val_samples']}, 特征{result['feature_count']}个"
                        train_status.style("color: #3fb950")
                        _refresh_models()
                    else:
                        train_status.text = f"❌ 训练失败: {result.get('error', '未知错误')}"
                        train_status.style("color: #f85149")
                except Exception as e:
                    train_progress.value = 0
                    train_status.text = f"❌ 异常: {e}"
                    train_status.style("color: #f85149")

            def _refresh_models():
                models = svc.list_saved_models()
                for m in models:
                    m["size_mb"] = f"{m['size_mb']:.2f}"
                model_table.rows = models

            _refresh_models()

        with ui.tab_panel("schedule"):
            with ui.card().classes("full-width"):
                ui.label("盘后自动因子挖掘").classes("text-h6 q-mb-md").style("color: #58a6ff")
                with ui.row().classes("full-width"):
                    sched_enabled = ui.switch(text="启用定时任务", value=config.alpha.schedule.enabled).classes("q-mr-md")
                    start_input = ui.input(label="开始时间", value=config.alpha.schedule.start_time).classes("q-mr-md")
                    end_input = ui.input(label="结束时间", value=config.alpha.schedule.end_time).classes("q-mr-md")
                days_select = ui.select(
                    label="运行日",
                    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    multiple=True,
                    value=config.alpha.schedule.days,
                ).classes("full-width q-mt-md")
                with ui.row().classes("q-mt-md"):
                    ui.button("▶ 启动", on_click=lambda: _start_schedule(), color="primary")
                    ui.button("⏸ 暂停", on_click=lambda: _pause_schedule(), color="warning")
                    ui.button("▶ 恢复", on_click=lambda: _resume_schedule(), color="secondary")
                    ui.button("⏹ 停止", on_click=lambda: _stop_schedule(), color="negative")
                sched_status = ui.label("").classes("text-body2 q-mt-md").style("color: #8b949e")

            def _start_schedule():
                result = svc.start_schedule()
                sched_status.text = f"调度器已启动: {result['status']}"
                sched_status.style("color: #3fb950")

            def _pause_schedule():
                svc.pause_schedule()
                sched_status.text = "调度器已暂停"
                sched_status.style("color: #d29922")

            def _resume_schedule():
                svc.resume_schedule()
                sched_status.text = "调度器已恢复"
                sched_status.style("color: #3fb950")

            def _stop_schedule():
                svc.stop_schedule()
                sched_status.text = "调度器已停止"
                sched_status.style("color: #f85149")

        with ui.tab_panel("operators"):
            ui.label("ETF 专属算子参考").classes("text-h6 q-mb-md").style("color: #58a6ff")
            operators = svc.get_operators()
            ui.table(
                columns=[
                    {"name": "op", "label": "算子", "field": "op", "align": "left"},
                    {"name": "name", "label": "名称", "field": "name", "align": "left"},
                    {"name": "desc", "label": "说明", "field": "desc", "align": "left"},
                ],
                rows=operators,
            ).classes("full-width")
