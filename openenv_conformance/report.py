"""Findings, A-F grading and output renderers (console / json / markdown / badge)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from urllib.parse import quote

SEVERITIES = ("critical", "high", "medium", "low")
PENALTY = {"critical": 40, "high": 20, "medium": 10, "low": 3}
GRADE_ORDER = "ABCDF"
BADGE_COLOR = {"A": "brightgreen", "B": "green", "C": "yellow", "D": "orange", "F": "red"}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    resource: str = ""
    fix: str = ""


@dataclass
class Report:
    tool: str
    target: str
    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def score(self) -> float:
        # Each distinct rule costs its severity penalty once; repeats of the
        # same rule add 25% each, capped at 2x, so one systemic gap does not
        # swamp everything else in a large manifest set.
        per_rule: Dict[tuple, int] = {}
        for f in self.findings:
            key = (f.rule, f.severity)
            per_rule[key] = per_rule.get(key, 0) + 1
        s = 100.0
        for (_, sev), n in per_rule.items():
            s -= PENALTY.get(sev, 0) * min(1 + 0.25 * (n - 1), 2.0)
        return max(0.0, s)

    @property
    def grade(self) -> str:
        s = self.score
        g = "A" if s >= 90 else "B" if s >= 80 else "C" if s >= 70 else "D" if s >= 60 else "F"
        if any(f.severity == "critical" for f in self.findings) and g in "ABC":
            g = "D"  # any critical caps the grade at D
        return g

    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in SEVERITIES}
        for f in self.findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "target": self.target,
            "grade": self.grade,
            "score": round(self.score, 1),
            "counts": self.counts(),
            "stats": self.stats,
            "findings": [asdict(f) for f in self.findings],
        }


def grade_at_least(grade: str, minimum: str) -> bool:
    return GRADE_ORDER.index(grade) <= GRADE_ORDER.index(minimum.upper())


def severity_at_least(sev: str, threshold: str) -> bool:
    return SEVERITIES.index(sev) <= SEVERITIES.index(threshold)


def render(report: Report, fmt: str, label: Optional[str] = None) -> str:
    label = label or report.tool
    if fmt == "json":
        return json.dumps(report.to_dict(), indent=2)
    if fmt == "badge":
        return (
            f"https://img.shields.io/badge/{quote(label.replace('-', '--'))}-"
            f"{report.grade}%20({report.score:.0f})-{BADGE_COLOR[report.grade]}"
        )
    if fmt == "badge-endpoint":
        return json.dumps(
            {
                "schemaVersion": 1,
                "label": label,
                "message": f"{report.grade} ({report.score:.0f})",
                "color": BADGE_COLOR[report.grade],
            }
        )
    order = {s: i for i, s in enumerate(SEVERITIES)}
    items = sorted(report.findings, key=lambda f: (order[f.severity], f.rule, f.resource))
    c = report.counts()
    if fmt == "markdown":
        out = [
            f"## {report.tool}: `{report.target}`",
            "",
            f"**Grade {report.grade}** ({report.score:.0f}/100) | "
            + " | ".join(f"{k}: {v}" for k, v in c.items()),
            "",
        ]
        if report.stats:
            out.append(" ".join(f"`{k}={v}`" for k, v in report.stats.items()))
            out.append("")
        if not items:
            out.append("No findings.")
        else:
            out += ["| Severity | Rule | Resource | Message | Fix |", "|---|---|---|---|---|"]
            for f in items:
                msg = f.message.replace("|", "\\|")
                fix = f.fix.replace("|", "\\|")
                out.append(f"| {f.severity} | `{f.rule}` | {f.resource} | {msg} | {fix} |")
        return "\n".join(out)
    # console
    out = [f"{report.tool}  {report.target}", ""]
    for f in items:
        out.append(f"  [{f.severity.upper():8}] {f.rule:30} {f.resource}")
        out.append(f"             {f.message}")
        if f.fix:
            out.append(f"             fix: {f.fix}")
    if not items:
        out.append("  no findings")
    out.append("")
    if report.stats:
        out.append("  " + "  ".join(f"{k}={v}" for k, v in report.stats.items()))
    out.append(
        f"  Grade {report.grade}  score {report.score:.0f}/100  "
        + "  ".join(f"{k}={v}" for k, v in c.items())
    )
    return "\n".join(out)
