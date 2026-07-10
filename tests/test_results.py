import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / "raw" / name).read_text())


def test_raw_summary_counts_are_consistent() -> None:
    for path in (RESULTS / "raw").glob("*.json"):
        data = json.loads(path.read_text())
        if "results" not in data:
            continue
        assert data["attempts"] == len(data["results"])
        on_pad_count = data.get("on_pad_count", data.get("broad_on_pad"))
        strict_count = data.get("pad_center_3cm_lift_count", data.get("strict_center_lift"))
        assert on_pad_count == sum(bool(row["on_pad"]) for row in data["results"])
        assert strict_count == sum(
            bool(row["pad_center_3cm_lift"]) for row in data["results"]
        )


def test_n100_comparisons_use_identical_episode_order() -> None:
    baseline = _load("sft_distilled_gaussian_ode_n100.json")["episode_ids"]
    for name in (
        "rl_stage1_gaussian_ode_n100.json",
        "rl_stage2_gaussian_ode_n100_b20.json",
        "rl_stage2_gaussian_ode_n100_b1.json",
        "rl_stage2_zero_ode_n100.json",
    ):
        assert _load(name)["episode_ids"] == baseline


def test_compact_index_matches_raw_counts() -> None:
    index = json.loads((RESULTS / "results.json").read_text())
    for experiment in index["experiments"]:
        raw = json.loads((RESULTS / experiment["raw"]).read_text())
        assert experiment["attempts"] == raw["attempts"]
        assert experiment["on_pad"] == raw["on_pad_count"]
        assert experiment["pad_center_3cm_lift"] == raw["pad_center_3cm_lift_count"]
