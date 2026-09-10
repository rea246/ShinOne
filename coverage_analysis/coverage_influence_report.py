"""Reference-centered kNN votes, fixed-radius reach and offline heatmap report.

Uses the existing projected CSVs; never fits a frame or changes R by group.
Run with --help, or call generate_report after coverage_domain_kpca.run().
"""
import argparse
import base64
from collections import Counter
from html import escape
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from .group_reinforce_nnvote import _read_table, group_labels, nearest_votes
except ImportError:
    from group_reinforce_nnvote import _read_table, group_labels, nearest_votes

KP_COLS = ["KP1", "KP2", "KP3"]


def _finite_table(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if "influence_source_row" in frame.columns:
        raise ValueError(f"{label}: influence_source_row is reserved for output")
    good = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
    clean = frame.loc[good].copy()
    clean.insert(0, "influence_source_row", np.flatnonzero(good))
    if not len(clean):
        raise ValueError(f"{label}: no finite coordinates")
    return clean.reset_index(drop=True), int((~good).sum())


def analyze(reference, sample, radius, target=None, *, kp_cols=KP_COLS,
            gauge_col="gauge_name", group_words=None, group_delim="_", topk=3):
    """All decisions use all supplied KP axes, scaled by the FULL reference."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and >= 0")
    if len(kp_cols) < 2 or len(set(kp_cols)) != len(kp_cols):
        raise ValueError("Supply at least two distinct KP columns")
    ref, drop_r = _finite_table(reference, kp_cols, "Reference")
    sam, drop_s = _finite_table(sample, kp_cols, "Sample")
    if gauge_col not in sam or sam[gauge_col].isna().any() or not group_delim:
        raise ValueError(f"Sample requires nonmissing {gauge_col} and a nonempty group delimiter")
    if target is None:
        use = ref.copy()
        if "cover_status" in use:
            if not use.cover_status.isin(["covered", "gap", "out_of_domain"]).all():
                raise ValueError("Reference has invalid cover_status")
            use = use.loc[use.cover_status != "out_of_domain"].copy()
        drop_t = 0
        scope = "Reference HDR" if "cover_status" in ref else "Full reference"
    else:
        use, drop_t = _finite_table(target, kp_cols, "Target")
        # CSV filtering/re-export can change the last float bits. Membership alone
        # uses 10 decimal places; all distances and coverage keep full precision.
        available = Counter(map(tuple, ref[kp_cols].to_numpy(float).round(10)))
        requested = Counter(map(tuple, use[kp_cols].to_numpy(float).round(10)))
        if requested - available:
            raise ValueError("Target must be a subset of the reference scatter (including duplicate counts)")
        scope = "Explicit reference subset (all finite supplied target rows)"
    use = use.reset_index(drop=True)
    if not len(use):
        raise ValueError("No target rows remain in the selected reference domain")
    frames = []
    for label, frame in (("Reference", ref), ("Sample", sam), ("Target", use)):
        if "kpca_frame_id" in frame:
            ids = frame.kpca_frame_id.dropna().astype(str).unique()
            if len(ids) != 1 or frame.kpca_frame_id.isna().any():
                raise ValueError(f"{label}: invalid or mixed kpca_frame_id")
            frames.append(ids[0])
    if frames and (len(frames) != 3 or len(set(frames)) != 1):
        raise ValueError("Reference, sample and target must carry the same kpca_frame_id")
    scale = ref[kp_cols].to_numpy(float).std(axis=0)
    scale[scale == 0] = 1.0
    tn, sn = use[kp_cols].to_numpy(float) / scale, sam[kp_cols].to_numpy(float) / scale
    labels = group_labels(sam[gauge_col], group_words, group_delim)
    groups, codes = np.unique(labels.astype(str), return_inverse=True)
    print(f"[Influence] {len(use):,} targets, {len(sam):,} samples, {len(groups)} groups; fixed R={radius:.8g}", flush=True)
    dist, idx, weights = nearest_votes(tn, sn, topk)
    vote = np.zeros((len(tn), len(groups)))
    np.add.at(vote, (np.repeat(np.arange(len(tn)), idx.shape[1]), codes[idx.ravel()]), weights.ravel())
    # ponytail: O(targets * groups) storage; batch group diagnostics if thousands of groups are needed.
    distances = np.column_stack([cKDTree(sn[codes == g]).query(tn, k=1)[0] for g in range(len(groups))])
    hits = distances <= radius
    n_covering = hits.sum(axis=1)
    covered = n_covering > 0
    unique = hits & (n_covering[:, None] == 1)
    count = np.bincount(codes, minlength=len(groups))
    total = vote.sum(axis=0)
    ranking = pd.DataFrame({"group": groups, "n_patterns": count, "total_score": total,
        "score_pct": total / len(tn) * 100, "score_per_pattern": total / count,
        "coverage_pct": hits.mean(axis=0) * 100,
        "unique_loss_pp": unique.mean(axis=0) * 100,
        "shared_coverage_pct": (hits & ~unique).mean(axis=0) * 100,
        "mean_nn_distance": distances.mean(axis=0)})
    for name, metric in (("vote_rank", "total_score"), ("efficiency_rank", "score_per_pattern"),
                         ("coverage_rank", "coverage_pct"), ("unique_rank", "unique_loss_pp")):
        ranking[name] = ranking[metric].rank(method="min", ascending=False).astype(int)
    ranking = ranking.sort_values(["total_score", "group"], ascending=[False, True], kind="stable").reset_index(drop=True)
    union = np.zeros(len(tn), dtype=bool)
    gains, cumulative = [], []
    for group in ranking.group:
        mask = hits[:, np.flatnonzero(groups == group)[0]]
        gains.append(float((mask & ~union).mean() * 100))
        union |= mask
        cumulative.append(float(union.mean() * 100))
    ranking["gain_in_vote_order_pp"], ranking["cumulative_coverage_pct"] = gains, cumulative

    sample_score = np.bincount(idx.ravel(), weights=weights.ravel(), minlength=len(sam))
    nearest_two, nearest_two_idx = cKDTree(sn).query(tn, k=[1, 2])
    only_sample = (nearest_two[:, 0] <= radius) & (nearest_two[:, 1] > radius)
    sample_unique = np.bincount(nearest_two_idx[only_sample, 0], minlength=len(sam))
    sample_hits = cKDTree(tn).query_ball_point(sn, radius, return_length=True)
    patterns = sam.copy()
    patterns["influence_group"] = labels
    patterns["influence_score"] = sample_score
    patterns["influence_score_pct"] = sample_score / len(tn) * 100
    patterns["influence_coverage_pct"] = sample_hits / len(tn) * 100
    patterns["influence_unique_loss_pp"] = sample_unique / len(tn) * 100
    patterns["influence_rank"] = pd.Series(sample_score).rank(method="min", ascending=False).astype(int)
    patterns = patterns.sort_values(["influence_score", "influence_source_row"], ascending=[False, True])
    targets = use.copy()
    targets["influence_covered"] = covered
    targets["influence_nearest_distance"] = dist[:, 0]
    targets["influence_n_covering_groups"] = n_covering
    targets["influence_dominant_group"] = groups[vote.argmax(axis=1)]
    for r in range(idx.shape[1]):
        targets[f"influence_nn{r+1}_sample_row"] = sam.influence_source_row.to_numpy()[idx[:, r]]
        targets[f"influence_nn{r+1}_gauge"] = sam[gauge_col].astype(str).to_numpy()[idx[:, r]]
        targets[f"influence_nn{r+1}_group"] = labels[idx[:, r]]
        targets[f"influence_nn{r+1}_distance"] = dist[:, r]
        targets[f"influence_nn{r+1}_vote"] = weights[:, r]
    summary = {"schema_version": 1, "scope": scope, "n_reference": len(ref), "n_target": len(tn),
        "n_sample": len(sn), "n_groups": len(groups), "kp_cols": list(kp_cols), "ystd": scale.tolist(),
        "radius": float(radius), "topk_requested": topk, "topk_used": idx.shape[1],
        "coverage_pct": float(covered.mean() * 100), "n_gap": int((~covered).sum()),
        "dropped_nonfinite": {"reference": drop_r, "sample": drop_s, "target": drop_t},
        "frame_id": frames[0] if frames else None,
        "frame_check": "matching exported frame IDs" if frames else "legacy CSV: shared frame is not verifiable",
        "interpretation": "Geometric association in sampled reference; not validated physical performance.",
        "vote_policy": "R-free inverse-distance top-K, one vote per target including gaps; ties at K follow cKDTree.",
        "coverage_policy": "All KP axes; one fixed radius; denominator is selected target rows. Self hits included.",
        "group_words": group_words, "group_delimiter": group_delim, "gauge_column": gauge_col}
    return dict(reference=ref, target=targets, samples=patterns, ranking=ranking, groups=groups,
                hits=hits, unique=unique, votes=vote, distances=distances, summary=summary)


def heatmap_bins(result, bins):
    """REF-only shared extents; NaN for empty target bins; no 2D coverage test."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    columns = result["summary"]["kp_cols"]
    reference, target = result["reference"], result["target"]
    grids = []
    for a, b in combinations(columns[:3], 2):
        ref_count, xe, ye = np.histogram2d(reference[a], reference[b], bins=bins)
        def hist(values):
            return np.histogram2d(target[a], target[b], bins=(xe, ye), weights=values)[0]
        count = hist(None)
        def mean(values):
            return np.divide(hist(values), count, out=np.full_like(count, np.nan), where=count > 0)
        grids.append({"pair": (a, b), "x_edges": xe, "y_edges": ye, "ref_count": ref_count,
            "target_count": count, "coverage": mean(target.influence_covered.to_numpy(float)),
            "distance": mean(target.influence_nearest_distance),
            "group_coverage": [mean(result["hits"][:, g].astype(float)) for g in range(len(result["groups"]))],
            "group_unique": [mean(result["unique"][:, g].astype(float)) for g in range(len(result["groups"]))],
            "group_vote": [mean(result["votes"][:, g]) for g in range(len(result["groups"]))]})
    return grids


def write_bin_csv(grids, groups, output):
    rows = []
    for grid in grids:
        for x, y in np.ndindex(grid["target_count"].shape):
            base = {"pair": "/".join(grid["pair"]), "x_bin": x, "y_bin": y,
                "x_min": grid["x_edges"][x], "x_max": grid["x_edges"][x+1],
                "y_min": grid["y_edges"][y], "y_max": grid["y_edges"][y+1],
                "x_upper_inclusive": x == len(grid["x_edges"])-2,
                "y_upper_inclusive": y == len(grid["y_edges"])-2,
                "ref_count": int(grid["ref_count"][x, y]), "target_count": int(grid["target_count"][x, y])}
            rows.append({**base, "scope": "union", "group": "", "coverage_fraction": grid["coverage"][x, y],
                         "mean_distance": grid["distance"][x, y]})
            for g, name in enumerate(groups):
                rows.append({**base, "scope": "group", "group": name,
                    "coverage_fraction": grid["group_coverage"][g][x, y],
                    "unique_loss_fraction": grid["group_unique"][g][x, y],
                    "vote_share": grid["group_vote"][g][x, y]})
    pd.DataFrame(rows).to_csv(output, index=False)


def plot_report(result, grids, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    # Native Korean font is optional; report text remains readable on every platform.
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"):
        if family in installed:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e6e9ec")
    summary = result["summary"]
    def panels(filename, title, fields):
        fig, axes = plt.subplots(len(fields), len(grids), figsize=(5*len(grids)+1, 3.7*len(fields)),
                                 squeeze=False, constrained_layout=True)
        for row, (label, values, kind) in enumerate(fields):
            maximum = max(float(np.nanmax(v)) for v in values)
            norm = LogNorm(1, max(2, maximum)) if kind == "count" else Normalize(0, 1 if kind == "fraction" else max(maximum, 1e-12))
            for ax, grid, value in zip(axes[row], grids, values):
                plotted = np.where(value > 0, value, np.nan) if kind == "count" else value
                mesh = ax.pcolormesh(grid["x_edges"], grid["y_edges"], np.ma.masked_invalid(plotted.T),
                                     cmap=cmap, norm=norm, shading="flat", rasterized=True)
                a, b = grid["pair"]
                ax.set(xlabel=a, ylabel=b, title=f"{a} / {b}")
            opts = {"format": PercentFormatter(1)} if kind == "fraction" else {}
            fig.colorbar(mesh, ax=axes[row].tolist(), label=label, shrink=.85, **opts)
        fig.suptitle(title + "\nGray: no observations | All distance decisions use every selected KP axis", fontsize=12)
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
    panels("coverage_heatmap.png", f"Reference / target coverage | Fixed R={summary['radius']:.5g}", [
        ("Full REF count (log)", [g["ref_count"] for g in grids], "count"),
        ("Selected target count (log)", [g["target_count"] for g in grids], "count"),
        ("Covered / target in bin", [g["coverage"] for g in grids], "fraction"),
        ("Mean nearest sample distance", [g["distance"] for g in grids], "distance")])
    group_images = {}
    for i, name in enumerate(result["groups"]):
        filename = f"group_{i+1:03d}_heatmap.png"
        panels(filename, f"Group: {name} | Fixed R={summary['radius']:.5g}", [
            ("Group coverage / target in bin", [g["group_coverage"][i] for g in grids], "fraction"),
            ("Coverage lost without group / target in bin", [g["group_unique"][i] for g in grids], "fraction"),
            ("Mean R-free kNN vote share", [g["group_vote"][i] for g in grids], "fraction")])
        group_images[str(name)] = filename
    ranking = result["ranking"]
    shown = ranking.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, .36*len(shown)+1)), constrained_layout=True)
    for ax, field, title in zip(axes, ["score_pct", "score_per_pattern", "unique_loss_pp"],
                               ["Total kNN vote share (%)", "kNN score per pattern", "Coverage lost on removal (pp)"]):
        ax.barh(shown.group, shown[field], color="#167d9a")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Group influence | Top 20 by total vote (all groups in CSV / HTML)")
    fig.savefig(output / "group_ranking.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    all_distances = [result["distances"].min(axis=1)]
    names = ["All samples"]
    for name in ranking.group.head(10):
        i = np.flatnonzero(result["groups"] == name)[0]
        all_distances.append(result["distances"][:, i])
        names.append(str(name))
    for values, name in zip(all_distances, names):
        d = np.sort(values)
        axes[0].step(np.r_[0, d], np.r_[0, np.arange(1, len(d)+1)/len(d)], where="post", label=name)
    axes[0].axvline(summary["radius"], linestyle="--", color="#ad482c", label="Fixed R")
    axes[0].set(xlabel="Allowed radius (REF-scaled KP)", ylabel="Target coverage", ylim=(0, 1.02), title="Coverage vs radius (top 10 groups)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].step(np.arange(len(ranking)+1), np.r_[0, ranking.cumulative_coverage_pct], where="post", color="#167d9a")
    axes[1].set(xlabel="Groups added in total-vote rank order", ylabel="Target coverage (%)", ylim=(0, 102), title="Cumulative coverage at fixed R")
    fig.savefig(output / "coverage_curves.png", dpi=130)
    plt.close(fig)
    return group_images


def write_html(result, output, group_images, title):
    summary = result["summary"]
    def picture(filename, alt):
        encoded = base64.b64encode((output / filename).read_bytes()).decode("ascii")
        return f'<img alt="{escape(alt, quote=True)}" src="data:image/png;base64,{encoded}">'
    def table(frame):
        return '<div class="table-wrap">' + frame.to_html(index=False, escape=True, border=0, float_format=lambda v: f"{v:.5g}") + '</div>'
    cards = "".join(f'<div class="card"><span>{label}</span><strong>{value}</strong></div>' for label, value in [
        ("분석 대상 REF", f"{summary['n_target']:,}"), ("Sample 패턴", f"{summary['n_sample']:,}"),
        ("집단", summary["n_groups"]), ("전체 coverage", f"{summary['coverage_pct']:.2f}%"),
        ("미커버 REF", f"{summary['n_gap']:,}"), ("고정 반경 R", f"{summary['radius']:.6g}")])
    options, sections = [], []
    for i, name in enumerate(result["ranking"].group):
        options.append(f'<option value="group-{i}">{escape(str(name))}</option>')
        row = result["ranking"].loc[result["ranking"].group == name].iloc[0]
        sections.append(f'<div class="group-panel" id="group-{i}" {"hidden" if i else ""}>'
            f'<p>집단 <b>{escape(str(name))}</b> · 단독 coverage {row.coverage_pct:.2f}% · '
            f'제거 시 손실 {row.unique_loss_pp:.2f}pp · 중복 coverage {row.shared_coverage_pct:.2f}%</p>'
            + picture(group_images[str(name)], f"{name} coverage, removal loss and vote maps") + '</div>')
    sample_columns = ["influence_rank", "influence_source_row", summary["gauge_column"], "influence_group",
        "influence_score", "influence_score_pct", "influence_coverage_pct", "influence_unique_loss_pp"]
    settings = escape(json.dumps(summary, indent=2, ensure_ascii=False))
    html = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference coverage influence</title>
<style>body{margin:0;background:#f1f5f7;color:#193444;font:15px/1.65 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:32px}h1{font-size:32px;margin:0}h2{font-size:23px}
section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.card{padding:18px;background:#193f53;color:white;border-radius:10px}.card span{display:block;font-size:13px}
.card strong{font-size:26px}.table-wrap{overflow:auto;max-height:560px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}
th,td{text-align:right;padding:9px;border-bottom:1px solid #dce5e9}th{position:sticky;top:0;background:#e7f0f4}
th button{font:inherit;border:0;background:none;cursor:pointer;color:inherit}select{padding:10px;min-width:240px;font:inherit}
.note{color:#536d7b}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#126582}
@media(max-width:700px){main{padding:12px}section{padding:12px}h1{font-size:25px}}
</style></head><body><main>'''
    html += f'<h1>{escape(title)}</h1><p class="note">B Case · Reference 기준 sample / 집단의 기하학적 연관도</p>{cards}'
    html += '''<section><h2>점수를 읽는 기준</h2><p>각 target REF는 가장 가까운 sample top-K에 역거리 비율로 총 1점을 분배합니다.
<b>투표점수는 반경과 무관하며 미커버 REF도 투표합니다.</b> 반면 coverage는 모든 선택 KP 축에서 최근접 거리가 고정 R 이하인 REF의 비율입니다.</p>
<p>total_score / score_pct는 총 연관도, score_per_pattern은 그룹 크기를 나눈 패턴당 점수입니다.
unique_loss_pp는 해당 집단을 모두 제거했을 때 전체 coverage가 줄어드는 퍼센트포인트입니다.
shared_coverage_pct는 다른 집단도 덮는 영역입니다. gain_in_vote_order_pp는 총점 순서로 추가할 때 새로 덮는 비율이며 순서에 의존합니다.</p>
<p>Heatmap은 이 거리 판정을 집계한 그림입니다. 분모는 각 칸 안의 <b>선택된 target REF 수</b>이며 빈 칸은 회색입니다.
격자 경계와 정규화는 전체 reference로 고정합니다. 중복 좌표의 행도 각각 세며 sample과의 자기 일치도 포함합니다.
2D에서 겹쳐 보이는 것만으로 covered 판정을 하지 않습니다. 기하학 점수는 Mask/OPC 성능 기여도로 검증된 값이 아닙니다.</p></section>'''
    html += '<section><h2>Reference의 어느 영역을 덮는가</h2>' + picture("coverage_heatmap.png", "Reference density, target density, coverage and nearest distance") + '</section>'
    html += '<section><h2>집단 순위와 중복</h2><p>표 제목을 누르면 정렬합니다. 동점은 같은 순위이며 누적 곡선의 동점 순서는 집단명 순입니다.</p>' + picture("group_ranking.png", "Group ranking") + table(result["ranking"]) + '</section>'
    html += '<section><h2>집단별 coverage / 제거 시 손실 / 투표 분포</h2><label for="group-select">집단 선택 </label><select id="group-select">' + ''.join(options) + '</select>' + ''.join(sections) + '</section>'
    html += '<section><h2>반경과 집단 추가에 따른 변화</h2>' + picture("coverage_curves.png", "Radius sensitivity and cumulative coverage") + '</section>'
    html += '<section><h2>Sample 패턴 순위</h2><p>총점 상위 100개를 표시합니다. 전체 패턴과 원본 열은 sample_influence_ranking.csv에 있습니다. row는 입력 scatter의 0-based 데이터 행번호입니다.</p>' + table(result["samples"][sample_columns].head(100)) + '</section>'
    html += f'<section><h2>재현 조건과 데이터</h2><p>{escape(summary["scope"])} · {escape(summary["frame_check"])}</p>'
    html += '<p>Legacy CSV에는 프레임 확인 정보가 없습니다. 같은 Reference cache에서 생성된 입력인지 확인해야 합니다.</p><p>그림은 HTML에 내장되어 인터넷 없이 볼 수 있습니다. CSV는 이 HTML과 같은 폴더에 저장됩니다.</p><ul>'
    for filename in ("group_influence_ranking.csv", "sample_influence_ranking.csv", "reference_influence.csv", "coverage_influence_bins.csv", "coverage_influence_summary.json"):
        html += f'<li><a href="{filename}">{filename}</a></li>'
    html += f'</ul><details><summary>전체 분석 조건</summary><pre>{settings}</pre></details></section>'
    html += '''<script>
document.getElementById('group-select').addEventListener('change', e => {
 document.querySelectorAll('.group-panel').forEach(p => {p.hidden = p.id !== e.target.value;});
});
document.querySelectorAll('th').forEach(th => {
 const button = document.createElement('button'); button.textContent = th.textContent + ' ↕';
 th.textContent = ''; th.appendChild(button);
 button.addEventListener('click', () => {
  const table = th.closest('table'), body = table.tBodies[0], col = th.cellIndex;
  const asc = th.dataset.asc !== 'true'; th.dataset.asc = String(asc);
  const rows = Array.from(body.rows);
  rows.sort((a,b) => {const x=a.cells[col].textContent, y=b.cells[col].textContent;
   const nx=Number(x), ny=Number(y); const d=Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y);
   return asc?d:-d;}); rows.forEach(r => body.appendChild(r));
 });
});
</script></main></body></html>'''
    (output / "coverage_influence_report.html").write_text(html, encoding="utf-8")


def generate_report(reference, sample, radius, output, target=None, *, bins=24,
                    title="Reference coverage & group influence", provenance=None, **kwargs):
    result = analyze(reference, sample, radius, target, **kwargs)
    grids = heatmap_bins(result, bins)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result["summary"].update(bins=bins, provenance=provenance or {})
    for name, key in (("group_influence_ranking", "ranking"), ("sample_influence_ranking", "samples"),
                      ("reference_influence", "target")):
        result[key].to_csv(output / f"{name}.csv", index=False)
    write_bin_csv(grids, result["groups"], output / "coverage_influence_bins.csv")
    (output / "coverage_influence_summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    images = plot_report(result, grids, output)
    write_html(result, output, images, title)
    print(f"[Influence] Saved {output / 'coverage_influence_report.html'}", flush=True)
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path, help="Full kpca_reference_scatter.csv; always required for scale")
    parser.add_argument("--sample", required=True, type=Path, help="kpca_sample_scatter.csv from the same reference frame")
    parser.add_argument("--target", type=Path, help="Optional filtered reference CSV/TSV; all finite supplied rows are targets")
    parser.add_argument("--summary", type=Path, help="Matching kpca_summary_metrics.json (defaults beside --sample)")
    parser.add_argument("--radius", type=float, help="Fixed radius; otherwise read from --summary")
    parser.add_argument("--group-col", default="gauge_name")
    parser.add_argument("--group-delim", default="_")
    parser.add_argument("--group-words", nargs="+", help="Ordered substring groups; first match wins, unmatched = other")
    parser.add_argument("--kp-cols", nargs="+", default=KP_COLS)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=Path("coverage_plots/influence"))
    args = parser.parse_args()
    reference, sample = _read_table(args.ref), _read_table(args.sample)
    target = _read_table(args.target) if args.target else None
    radius, radius_source = args.radius, "explicit --radius"
    if radius is None:
        path = args.summary or args.sample.with_name("kpca_summary_metrics.json")
        if not path.is_file():
            parser.error("Supply --radius or the matching --summary; a group-specific radius is never inferred")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("kpca_frame_id"):
            for frame in (reference, sample):
                if "kpca_frame_id" not in frame or not frame.kpca_frame_id.eq(summary["kpca_frame_id"]).all():
                    parser.error("Summary frame ID differs from the input scatters")
        if args.kp_cols != summary.get("kp_cols", KP_COLS):
            parser.error("Summary radius uses different KP axes; supply an explicit radius for the selected axes")
        radius = summary["repr_radius_kpca"]
        radius_source = str(path.resolve())
    generate_report(reference, sample, radius, args.out_dir, target, bins=args.bins,
        gauge_col=args.group_col, group_words=args.group_words, group_delim=args.group_delim,
        topk=args.topk, kp_cols=args.kp_cols,
        provenance={"reference": str(args.ref.resolve()), "sample": str(args.sample.resolve()),
                    "target": str(args.target.resolve()) if args.target else None, "radius_source": radius_source})


if __name__ == "__main__":
    cli()
