import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import METHODS
from .io_utils import atomic_json

LABELS = {"diffusion": "Diffusion (cosine VP)", "linear": "Linear FM",
          "linear_extra": "Linear FM + extra", "reflow": "Reflow"}
COLORS = dict(zip(METHODS, ("#3b72b9", "#d98a22", "#7c69b0", "#209b8b")))


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_rows(rows):
    """Adaptive reference NFEs are metadata, not different experimental groups."""
    grouped = defaultdict(list)
    for r in rows:
        key_nfe = "adaptive_reference" if r["sampler"] == "heun_reference" else r["nfe"]
        grouped[(r["task"], r["method"], r["sampler"], key_nfe)].append(r)
    summary = []
    for key, group in grouped.items():
        item = dict(zip(("task", "method", "sampler", "nfe"), key))
        item["seed_count"] = len(group)
        item["actual_nfe_min"] = min(r["nfe"] for r in group)
        item["actual_nfe_max"] = max(r["nfe"] for r in group)
        item["reference_all_converged"] = all(r["reference_converged"] for r in group)
        item["reference_failed_seeds"] = sum(not r["reference_converged"] for r in group)
        for metric in ("sw2", "solver_mse", "mode_coverage", "valid_fraction",
                       "straightness", "normalized_straightness"):
            values = [r[metric] for r in group if metric in r]
            if values:
                item[metric + "_mean"], item[metric + "_std"] = mean_std(values)
        summary.append(item)
    return summary


def build_report(out):
    out = Path(out)
    report = out / "report"
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    smoke = bool(config.get("synthetic_fashion")) or "SMOKE" in config["name"]
    prefix = "SMOKE TEST — NOT SCIENTIFIC RESULTS | " if smoke else ""
    rows, evaluations = [], {}
    for file in sorted(out.glob("*/seed_*/*/evaluation.json")):
        value = json.loads(file.read_text(encoding="utf-8"))
        rows.extend(value["rows"])
        row = value["rows"][0]
        evaluations[(row["task"], row["seed"], row["method"])] = value
    if not rows:
        raise RuntimeError("No completed evaluations found")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (report / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})
    summary = aggregate_rows(rows)
    with (report / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(s.keys() for s in summary))))
        writer.writeheader()
        writer.writerows(summary)
    atomic_json(report / "summary.json", summary)
    baselines = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(out.glob("toy/seed_*/data_baseline.json"))]
    atomic_json(report / "data_baselines.json", baselines)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    figures = []

    def save(fig, name, caption):
        fig.savefig(report / f"{name}.png", bbox_inches="tight")
        fig.savefig(report / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        figures.append((name, caption))

    def curve(task, metric, ylabel, name, log_y=False):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in METHODS:
            subset = [r for r in rows if r["task"] == task and r["method"] == method
                      and r["sampler"] == "euler" and metric in r]
            if not subset:
                continue
            xs = sorted({r["nfe"] for r in subset})
            ms = [mean_std([r[metric] for r in subset if r["nfe"] == x]) for x in xs]
            means, stds = np.array(ms).T
            if log_y:
                means = np.maximum(means, 1e-12)
            provisional = metric == "solver_mse" and any(not r["reference_converged"] for r in subset)
            ax.plot(xs, means, "o--" if provisional else "o-",
                    label=LABELS[method] + (" [ref unverified]" if provisional else ""),
                    color=COLORS[method])
            if len({r["seed"] for r in subset}) > 1:
                lower = np.maximum(means - stds, 1e-12 if log_y else 0)
                ax.fill_between(xs, lower, means + stds, alpha=.15, color=COLORS[method])
        if task == "toy" and metric == "sw2" and baselines:
            baseline_mean = float(np.mean([b["mean"] for b in baselines]))
            ax.axhline(baseline_mean, color="#606870", linestyle=":",
                       label="Data-data MC mean (not a bound)")
        if task == "toy" and metric == "valid_fraction":
            ax.axhline(1-np.exp(-4.5), color="#606870", linestyle=":", label="Target expectation")
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set(xlabel="Network function evaluations (Euler)", ylabel=ylabel,
               title=prefix + task + ": mean +/- seed SD")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, name, f"{task}：{ylabel} 随 NFE 的变化；阴影为种子间样本标准差，单种子不画阴影。")

    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        curve(task, "solver_mse", "Endpoint MSE vs own reference", f"{task}_solver_error", True)
        if task == "toy":
            curve(task, "sw2", "Empirical sliced Wasserstein-2", "toy_distribution_quality")
            curve(task, "mode_coverage", "Covered modes (out of 8)", "toy_mode_coverage")
            curve(task, "valid_fraction", "Fraction within 3 sigma of a mode", "toy_valid_fraction")
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, method in enumerate(METHODS):
            values = [r["normalized_straightness"] for r in rows
                      if r["task"] == task and r["method"] == method
                      and r["sampler"] == "heun_reference"]
            if values:
                m, sd = mean_std(values)
                ax.bar(i, m, yerr=sd if len(values) > 1 else None, capsize=4, color=COLORS[method])
                ax.scatter(np.full(len(values), i), values, c="black", s=12, zorder=3)
        ax.set_xticks(range(4), [LABELS[m] for m in METHODS], rotation=12)
        ax.set_ylabel("S / (E[Z0^2] + E[Z1^2])")
        ax.set_title(prefix + task + ": actual ODE trajectory")
        fig.tight_layout()
        save(fig, f"{task}_straightness", "实际 ODE 轨迹的速度变化；不是训练插值的直线程度。归一化规则见方法说明。")

        seed = min(r["seed"] for r in rows if r["task"] == task)
        arrays = {}
        for method in METHODS:
            path = out / task / f"seed_{seed}" / method / "samples.npz"
            if path.exists():
                with np.load(path) as data:
                    arrays[method] = {key: data[key] for key in data.files}
        nfes = config["tasks"][task]["nfes"]
        if task == "toy":
            fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), squeeze=False)
            for ax, method in zip(axes[0], METHODS):
                if method not in arrays:
                    continue
                data = arrays[method]
                ax.scatter(*data["target"][:512].T, s=5, c="#d3d9e0", alpha=.6)
                trace = data["trace"]
                for j in range(min(24, trace.shape[1])):
                    ax.plot(trace[:, j, 0], trace[:, j, 1], lw=.8, alpha=.8)
                ax.scatter(*trace[0].T, s=9, c="black", marker="x")
                ax.set(title=LABELS[method], xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            fig.suptitle(prefix + f"Actual ODE trajectories, seed {seed}; same initial noise")
            fig.tight_layout()
            save(fig, "toy_trajectories", "固定首个种子、固定初始噪声的实际轨迹；灰点为目标分布。")
            columns = [f"nfe_{n}" for n in nfes] + ["reference"]
            fig, axes = plt.subplots(4, len(columns), figsize=(3 * len(columns), 11), squeeze=False)
            for i, method in enumerate(METHODS):
                for j, col in enumerate(columns):
                    ax = axes[i, j]
                    if method in arrays:
                        x = arrays[method][col]
                        ax.scatter(*x.T, s=2, color=COLORS[method], alpha=.5)
                    ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
                    if i == 0:
                        ax.set_title(col.replace("_", " "))
                    if j == 0:
                        ax.set_ylabel(LABELS[method])
            fig.suptitle(prefix + "Toy generated distributions (fixed plotting window)")
            fig.tight_layout()
            save(fig, "toy_distributions", "二维生成分布；显示窗口固定为 [-4,4]，窗口外离群点仍计入定量指标。")
        else:
            columns = [f"nfe_{n}" for n in nfes] + ["reference"]
            fig, axes = plt.subplots(4, len(columns), figsize=(3 * len(columns), 11), squeeze=False)
            clipping = []
            for i, method in enumerate(METHODS):
                for j, col in enumerate(columns):
                    ax = axes[i, j]
                    if method in arrays:
                        raw = arrays[method][col][:16, 0]
                        clipping.append((method, col, float(np.mean(np.abs(raw) > 1))))
                        grid_rows = max(1, (len(raw) + 3) // 4)
                        tiles = np.full((grid_rows * 28, 4 * 28), 0.5)
                        for k, image in enumerate(raw):
                            tiles[(k // 4) * 28:(k // 4 + 1) * 28,
                                  (k % 4) * 28:(k % 4 + 1) * 28] = np.clip((image + 1) / 2, 0, 1)
                        ax.imshow(tiles, cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(col.replace("_", " "))
                    if j == 0:
                        ax.set_ylabel(LABELS[method])
            fig.suptitle(prefix + f"Fashion pilot, seed {seed}; up to 16 fixed noises, display clipped")
            fig.tight_layout()
            save(fig, "fashion_samples", "网格仅展示首个种子的前 16 个固定噪声；定量曲线按全部种子汇总。仅显示时裁剪，求解及指标不裁剪。")
            atomic_json(report / "fashion_display_clipping.json", clipping)

    # Per-path oracle error is diagnostic; absolute scales differ by path.
    fig, ax = plt.subplots(figsize=(6.4, 4))
    has_field = False
    for method in METHODS[:-1]:
        fs = [v["field_error"] for (t, _, m), v in evaluations.items()
              if t == "toy" and m == method and v["field_error"] is not None]
        if fs:
            has_field = True
            y = np.array([v["mse"] for v in fs])
            ax.plot(fs[0]["times"], y.mean(0), label=LABELS[method], color=COLORS[method])
    if has_field:
        ax.set(xlabel="Generation time", ylabel="MSE against own exact marginal field",
               title=prefix + "Toy oracle diagnostic (path-dependent scale)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, "toy_oracle_field_error", "独立配对组相对自身解析边缘向量场的误差；不同路径速度尺度不同，不能仅凭原始 MSE 排名。reflow 无该解析参照，故不展示。")
    else:
        plt.close(fig)

    failed_refs = [
        f"{task}/seed_{seed}/{method}"
        for (task, seed, method), val in evaluations.items()
        if not val["reference_check"]["converged"]]
    teacher_warnings = []
    for p in out.glob("*/seed_*/pairs/teacher_solver.json"):
        if not json.loads(p.read_text(encoding="utf-8"))["converged"]:
            teacher_warnings.append(str(p.relative_to(out)))
    warnings = []
    if smoke:
        warnings.append("这是极小规模合成数据烟雾测试，不能作为组会实验结果。")
    if failed_refs:
        warnings.append("以下参考求解器未通过步长加倍收敛检查，求解误差结论须谨慎：" + "；".join(failed_refs))
    if teacher_warnings:
        warnings.append("以下教师求解未通过收敛检查：" + "；".join(teacher_warnings))
    quality = dict(failed_reference_count=len(failed_refs)+len(teacher_warnings),
                   failed_evaluations=failed_refs, failed_teachers=teacher_warnings,
                   evaluation_count=len(evaluations), precision="fp32_no_tf32",
                   checks={f"{t}/seed_{s}/{m}": v["reference_check"]
                           for (t,s,m),v in evaluations.items()})
    atomic_json(report / "quality_checks.json", quality)
    if "fashion" in config["tasks"] and len(config["tasks"]["fashion"]["seeds"]) == 1:
        warnings.append("Fashion-MNIST 只有一个种子、短预算；仅提供初步图像观察，不能宣称稳定性能优势。")
    warnings.append("本实验不计算 FID；Fashion 图像部分没有独立的定量分布质量指标，straightness/求解误差不能替代生成质量。")
    lines = [
        "# 路径比较实验：自动生成的组会材料",
        "",
        "**实验状态：" + ("仅烟雾测试，非科研结果" if smoke else "实验已执行；结论须结合预算与收敛检查") + "**",
        "",
        "## 研究问题",
        "扩散插值、线性插值与 reflow 配对如何影响实际 ODE 轨迹和低 NFE 求解？",
        "",
        "## 方法与控制",
        "- 时间从噪声 0 到数据 1；所有方法使用同一个网络结构、速度 MSE、均匀时间采样。",
        "- 扩散组为余弦 VP 条件路径：X_t=sin(pi*t/2)*x+cos(pi*t/2)*epsilon。不是原始离散 DDPM 完整复现。",
        "- 线性 FM 使用独立配对；reflow 使用线性 FM 的 EMA 教师生成的端点配对。",
        "- linear_extra 与 reflow 从同一模型、EMA 和优化器状态分叉，额外更新数相同；教师生成成本单独记录。",
        "- 所有评估及教师生成采用 FP32 并关闭 TF32；训练仍保留原来的 BF16/TF32 配置。",
        "- Euler NFE=N，Heun NFE=2N，RK4 NFE=4N；每个模型使用自己的参考解。",
        "- 参考精度要求连续两次 Heun 步长加倍检查通过，再与相近 NFE 的 RK4 解交叉核对。",
        "- 检查条件为 RMSE <= atol + rtol * max(RMS(reference),0.1)；这是有限探针诊断，不是严格误差界。",
        "- 不同种子的自适应参考 NFE 不再拆成不同组；summary.csv 单独记录实际 NFE 范围。",
        "- 原始 S 按坐标平均；归一化 S 除以 E[Z0^2]+E[Z1^2]+1e-8。",
        "- 二维 SW2 使用 128 个固定投影；模式覆盖要求距中心<=0.45，且该模式有效样本至少占总数 1%。",
        "- EMA 使用 warm-start decay=min(0.9999,(1+global_step)/(10+global_step))，避免短训练被初始权重主导。",
        "",
        "## 实际配置",
        "```json", json.dumps(config, indent=2, ensure_ascii=False), "```",
        "",
        "## 需要在汇报中明确的限制",
        *["- " + w for w in warnings],
        "",
        "## 数值结果",
        "完整求解诊断保存在 quality_checks.json；参考检查失败的求解误差曲线用虚线并标记 ref unverified。",
        "二维 SW2 的 data-data MC 水平线来自相同评估真值与独立真实抽样；它是有限样本尺度参考，不是理论下界或显著性检验。",
        "完整逐种子数据见 metrics.csv；均值和种子间样本标准差见 summary.csv。",
        "标准差不代表置信区间。单种子的 std=0 只是没有重复实验，不能解释为没有不确定性。",
        "",
    ]
    for task in tasks:
        lines.extend([f"### {task}：固定 NFE 对照", "",
                      "| 方法 | 种子数 | NFE | 求解 MSE，均值±SD | SW2，均值±SD |",
                      "|---|---:|---:|---:|---:|"])
        selected = 4 if 4 in config["tasks"][task]["nfes"] else config["tasks"][task]["nfes"][0]
        for method in METHODS:
            match = next((s for s in summary if s["task"] == task and s["method"] == method
                          and s["sampler"] == "euler" and s["nfe"] == selected), None)
            if match:
                err = f'{match["solver_mse_mean"]:.5g} ± {match["solver_mse_std"]:.3g}'
                sw = (f'{match["sw2_mean"]:.5g} ± {match["sw2_std"]:.3g}'
                      if "sw2_mean" in match else "未测")
                lines.append(f'| {LABELS[method]} | {match["seed_count"]} | {selected} | {err} | {sw} |')
        lines.append("")
    lines += [
        "## 六页汇报建议",
        "1. 提出问题：路径形状和端点配对怎样影响少步生成。",
        "2. 展示配置与四组对照，明确额外训练/教师成本。",
        "3. 展示二维实际轨迹与归一化 straightness。",
        "4. 同时展示目标分布质量、模式覆盖和求解误差曲线。",
        "5. 展示 Fashion 固定噪声样本与多种子曲线；图像网格只展示第一个种子。",
        "6. 根据实际数据写观察、可能解释和未排除的因素；不要预设 reflow 获胜。",
        "",
        "## 判读规则",
        "- 求解误差降低且分布质量保持，才支持更适合少步求解的解释。",
        "- 轨迹更直但模式覆盖下降，提示生成质量存在代价。",
        "- 高精度样本仍不好，应先检查学习误差与预算，而非只增加采样步数。",
        "- 不能用三组不同目标的训练 loss 原始数值直接比较学习能力。",
        "",
        "## 图表",
    ]
    for name, caption in figures:
        lines += ["", f"![{caption}]({name}.png)", "", caption]
    lines += [
        "", "## 来源",
        "- Flow Matching: https://arxiv.org/abs/2210.02747",
        "- Rectified Flow: https://arxiv.org/abs/2209.03003",
        "- RF 作者说明: https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html",
        "- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist",
    ]
    markdown = "\n".join(lines) + "\n"
    (report / "meeting_notes.md").write_text(markdown, encoding="utf-8")
    # Self-contained report: embedded PNGs, no local HTTP server or external JS.
    sections = [f"<h1>{'烟雾测试 · ' if smoke else ''}扩散路径 / 线性 FM / Reflow</h1>",
                "<p>自动生成的实验记录。下方数值均来自当前运行；不预设任何方法获胜。</p>",
                "<h2>汇报时必须说明</h2><ul>"
                + "".join("<li>" + html.escape(w) + "</li>" for w in warnings) + "</ul>"]
    for task in tasks:
        sections.append(f"<h2>{html.escape(task)}：定量摘要</h2>")
        sections.append("<table><tr><th>方法</th><th>采样器</th><th>NFE</th><th>种子数</th>"
                        "<th>求解 MSE 均值</th><th>SW2 均值</th></tr>")
        for s in summary:
            if s["task"] == task:
                sections.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in (
                    LABELS[s["method"]], s["sampler"], s["nfe"], s["seed_count"],
                    f'{s["solver_mse_mean"]:.5g}',
                    f'{s["sw2_mean"]:.5g}' if "sw2_mean" in s else "未测")) + "</tr>")
        sections.append("</table>")
    for name, caption in figures:
        encoded = base64.b64encode((report / f"{name}.png").read_bytes()).decode("ascii")
        sections.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
                        f"<figcaption>{html.escape(caption)}</figcaption></figure>")
    sections.append("<h2>方法、配置与汇报说明</h2><pre>"
                    + html.escape("\n".join(lines[:lines.index("## 图表")])) + "</pre>")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>路径比较实验报告</title><style>
body{font:16px/1.7 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 24px;color:#172b42}
h1,h2{color:#174e6b}figure{margin:32px 0}img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d5dfe8;padding:8px}
th{background:#eaf2f7}pre{white-space:pre-wrap;background:#f3f6fa;padding:20px;font-size:14px}
figcaption{color:#536477}li{margin:8px 0}@media print{figure{break-inside:avoid}body{margin:0}}
</style><body>""" + "\n".join(sections) + "</body></html>"
    (report / "report.html").write_text(page, encoding="utf-8")
    print(f"Report written: {report / 'report.html'}", flush=True)
    return {k: v for k,v in quality.items() if k != "checks"}
