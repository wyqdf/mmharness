from __future__ import annotations

import sys
from pathlib import Path

TEXT_CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEXT_CLASSIFICATION_DIR))

from meta_harness import _proposer_skill_dir


def test_s1_uses_official_skill() -> None:
    path = _proposer_skill_dir({"meta_meta": {"enabled": False}})

    assert path.name == "meta-harness"
    assert "Meta-Meta Extension" not in (path / "SKILL.md").read_text()


def test_s2_uses_vector_lineage_skill() -> None:
    path = _proposer_skill_dir(
        {
            "meta_meta": {
                "enabled": True,
                "vector_reward": True,
                "show_edges": True,
                "show_memory": False,
            }
        }
    )

    assert path.name == "meta-harness-s2"
    skill = (path / "SKILL.md").read_text()
    assert "S2 Vector-Lineage Harness Evolution" in skill
    assert "memory.summary" in skill


def test_s3_uses_meta_meta_skill() -> None:
    path = _proposer_skill_dir({"meta_meta": {"enabled": True, "show_memory": True}})

    assert path.name == "meta-harness-mm"
    assert "Meta-Meta Extension" in (path / "SKILL.md").read_text()
