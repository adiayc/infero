"""Aggregate case results into per-policy reports and a comparison table.

Runs the same eval set under several policies — each backend alone, plus the
router — and shows cost/quality/latency side by side, with savings and quality
delta against a baseline (usually the priciest backend-only policy).
"""

from __future__ import annotations

from pydantic import BaseModel

from infero.eval.runner import CaseResult


class PolicyReport(BaseModel):
    policy: str
    n: int
    # scorer name -> mean over the cases it applied to, None if it applied to
    # none.
    avg_quality: dict[str, float | None]
    total_cost_usd: float
    avg_latency_ms: float

    def primary_quality(self, prefer: str = "llm_judge") -> float | None:
        if prefer in self.avg_quality:
            return self.avg_quality[prefer]
        return next(iter(self.avg_quality.values()), None)


def aggregate(policy: str, results: list[CaseResult]) -> PolicyReport:
    """Average each scorer over the cases it actually scored.

    Cases a scorer skipped are excluded rather than counted as zero: an eval set
    seeded from telemetry has no golden answers, and averaging in zeros would
    report every model as scoring 0.000 on a dimension nobody measured.
    """

    n = len(results)
    scorer_names = {name for r in results for name in r.scores}
    avg_quality: dict[str, float | None] = {}
    for name in sorted(scorer_names):
        scored = [r.scores[name] for r in results if r.scores.get(name) is not None]
        avg_quality[name] = sum(scored) / len(scored) if scored else None

    return PolicyReport(
        policy=policy,
        n=n,
        avg_quality=avg_quality,
        total_cost_usd=sum(r.cost_usd for r in results),
        avg_latency_ms=sum(r.latency_ms for r in results) / n if n else 0.0,
    )


def render_markdown(reports: list[PolicyReport], baseline: str | None = None) -> str:
    if not reports:
        return "_no results_"

    scorers = sorted({name for r in reports for name in r.avg_quality})
    base = next((r for r in reports if r.policy == baseline), None)

    header = ["policy", "cases", *scorers, "total_cost", "avg_latency_ms"]
    if base is not None:
        header += ["cost_saved_vs_base", "quality_delta_vs_base"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    for r in reports:
        row = [
            r.policy,
            str(r.n),
            *[_score(r.avg_quality.get(s)) for s in scorers],
            f"${r.total_cost_usd:.4f}",
            f"{r.avg_latency_ms:.0f}",
        ]
        if base is not None:
            if base.total_cost_usd > 0:
                saved = (base.total_cost_usd - r.total_cost_usd) / base.total_cost_usd * 100
                row.append(f"{saved:+.1f}%")
            else:
                row.append("n/a")
            mine, theirs = r.primary_quality(), base.primary_quality()
            row.append(f"{mine - theirs:+.3f}" if mine is not None and theirs is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    footer = ""
    if any(v is None for r in reports for v in r.avg_quality.values()):
        footer = (
            "\n\n_“—” means that scorer had nothing to measure "
            "— e.g. cases with no reference answer._"
        )
    return "\n".join(lines) + footer


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
