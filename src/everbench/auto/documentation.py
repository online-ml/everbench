"""Human-readable research notes for the model's displayed source."""

from __future__ import annotations

import ast
from textwrap import fill
from typing import Any

from everbench.schema import AutoExperiment


def documented_source(
    source: str,
    metadata: dict[str, Any],
    experiments: list[AutoExperiment],
    counts: dict[str, int],
) -> str:
    """Prepend a live module docstring without changing the executable artifact."""
    generation = metadata.get("generation", 0)
    lines = [
        f"Autonomous model — generation {generation}.",
        "Generation 0 is the initial model; each promotion advances it by one.",
        "",
        "Research: "
        + ", ".join(f"{counts.get(status, 0):,} {status}" for status in ("promoted", "rejected", "failed", "running"))
        + ".",
        "",
        "Current champion:",
    ]

    def note(value: str, indent: str = "  ") -> None:
        lines.append(fill(value, width=88, initial_indent=indent, subsequent_indent=indent))

    hypothesis = metadata.get("hypothesis")
    note(hypothesis or "Initial model, pre-trained on mature historical observations; no promoted changes yet.")
    if not experiments:
        lines.extend(["", "No research rounds have run yet."])
    else:
        latest = experiments[0]
        lines.extend(["", "Research status:"])
        if latest.status == "running":
            note(
                f"{latest.researcher} is exploring candidates against generation {latest.parent_generation}. "
                "The selected hypothesis and sealed evaluation will appear when the round finishes."
            )
        else:
            note("No research round is currently running. The latest completed attempts are listed below.")
        lines.extend(["", f"Recent rounds (latest {len(experiments)}, newest first):"])
        for row in experiments:
            timestamp = row.started_at.strftime("%Y-%m-%d %H:%M %Z")
            transition = f"generation {row.parent_generation}"
            if row.status == "promoted":
                transition += f" → {row.parent_generation + 1}"
            lines.append(f"  {timestamp} — {row.status}, {transition}")
            if row.hypothesis:
                note(row.hypothesis, "    ")
            evaluation = row.evaluation or {}
            if evaluation:
                note(
                    f"{evaluation['metric']}: champion {evaluation['champion_score']:.6f}, "
                    f"candidate {evaluation['candidate_score']:.6f}; "
                    f"improvement {evaluation['improvement']:+.6f} "
                    f"on {evaluation['observations']:,} sealed observations.",
                    "    ",
                )
                failed = [item for item in evaluation.get("constraints", []) if not item["passed"]]
                for item in failed:
                    note(f"Failed constraint {item['name']}: {item['detail']}", "    ")
                if row.status == "rejected" and not failed:
                    note("Did not meet the promotion thresholds for improvement and evidence.", "    ")
            if row.error:
                note(f"Error: {row.error}", "    ")
        lines.extend(
            [
                "",
                "Scores compare candidates with their then-current champion on each round's",
                "sealed cohort; they are not live leaderboard scores. Positive improvement",
                "means better. A promoted hypothesis passed that round's promotion checks.",
            ]
        )
    module = ast.parse(source)
    original_docstring = ast.get_docstring(module)
    if original_docstring:
        lines.extend(["", "Model implementation:", original_docstring])
    # Escape arbitrary hypotheses/errors so the displayed header remains a valid
    # Python docstring, even when they contain quotes or backslashes.
    docstring = "\n".join(lines).replace("\\", "\\\\").replace('"', '\\"')
    header = f'"""{docstring}\n"""'
    if original_docstring is not None:
        node = module.body[0]
        assert node.end_lineno is not None and node.end_col_offset is not None
        source_lines = source.encode().splitlines(keepends=True)
        start = sum(map(len, source_lines[: node.lineno - 1])) + node.col_offset
        end = sum(map(len, source_lines[: node.end_lineno - 1])) + node.end_col_offset
        return (source.encode()[:start] + header.encode() + source.encode()[end:]).decode()
    return f"{header}\n\n{source}"
