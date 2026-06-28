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


def test_s3_uses_meta_meta_skill() -> None:
    path = _proposer_skill_dir({"meta_meta": {"enabled": True}})

    assert path.name == "meta-harness-mm"
    assert "Meta-Meta Extension" in (path / "SKILL.md").read_text()
