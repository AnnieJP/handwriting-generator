"""
Auto-updating test log writer.

Both `train_gan.py` and `train_diffusion.py` call `append_test_log` at the end
of a smoke-test run to record config + results in TEST_LOG.md.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union


DEFAULT_LOG = Path("TEST_LOG.md")

_HEADER = """# Test Log

Auto-updated by `train_gan.py --smoke_test` and `train_diffusion.py --smoke_test`.
Each entry records configuration, per-epoch loss, and verdict of one
pipeline-verification cycle.

**Pass thresholds:** loss ratio (end / start) below 0.70 → PASS,
0.70–1.00 → MARGINAL, ≥ 1.00 → FAIL.

---

"""


def compute_verdict(ratio: float) -> str:
    if ratio < 0.7:
        return "PASS"
    if ratio < 1.0:
        return "MARGINAL"
    return "FAIL"


def append_test_log(
    model_name: str,
    config: Dict[str, object],
    epoch_losses: List[float],
    wall_time_seconds: float,
    output_dir: Union[str, Path],
    aux_losses: Dict[str, List[float]] = None,
    log_path: Union[str, Path] = DEFAULT_LOG,
    notes: str = "",
) -> Path:
    """Append a smoke test result entry to a markdown log file.

    `epoch_losses` is the primary series used for verdict computation.
    `aux_losses` is an optional dict of additional series to log (e.g., for GAN
    we log D-loss as primary and G-loss as auxiliary).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        log_path.write_text(_HEADER)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(epoch_losses)
    start = epoch_losses[0] if epoch_losses else float("nan")
    end = epoch_losses[-1] if epoch_losses else float("nan")
    ratio = (end / start) if (epoch_losses and start) else float("inf")
    verdict = compute_verdict(ratio) if epoch_losses else "SKIPPED"

    lines = [
        f"## {timestamp} — {model_name.upper()} smoke test",
        "",
        "**Config:**",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for k, v in config.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines += [
        "",
        "**Results:**",
        "",
        f"- Epochs run: `{n}`",
        f"- Loss epoch 1: `{start:.4f}`",
        f"- Loss epoch {n}: `{end:.4f}`",
        f"- Ratio (end / start): `{ratio:.3f}`",
        f"- Wall time: `{wall_time_seconds:.1f}s`",
        f"- Output dir: `{output_dir}`",
        f"- **Verdict: {verdict}**",
        "",
        "**Per-epoch loss:** " + " → ".join(f"`{l:.4f}`" for l in epoch_losses),
        "",
    ]
    if aux_losses:
        for series_name, series_values in aux_losses.items():
            lines.append(
                f"**Per-epoch {series_name}:** "
                + " → ".join(f"`{v:.4f}`" for v in series_values)
            )
            lines.append("")
    if notes:
        lines += ["**Notes:** " + notes, ""]
    lines += ["---", "", ""]

    with open(log_path, "a") as f:
        f.write("\n".join(lines))

    return log_path
