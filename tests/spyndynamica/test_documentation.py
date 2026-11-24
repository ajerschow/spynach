"""The guides name everything the package exports."""

import re
from pathlib import Path

import spyndynamica as sd

DOCS = Path(__file__).resolve().parents[2] / "docs"
GUIDES = (
    "spyndynamica.md",
    "spyndynamica-simulations.md",
    "spyndynamica-symmetry.md",
    "spyndynamica-adiabatic.md",
    "coming-from-spindynamica.md",
)


def test_every_exported_name_is_in_a_guide():
    text = "".join((DOCS / name).read_text(encoding="utf-8") for name in GUIDES)
    missing = [name for name in sorted(sd.__all__) if not re.search(rf"\b{re.escape(name)}\b", text)]
    assert not missing, f"not in any guide: {missing}"
    assert len(set(sd.__all__)) == len(sd.__all__)
    assert all(hasattr(sd, name) for name in sd.__all__)


def test_guides_link_to_each_other():
    for name in GUIDES:
        text = (DOCS / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(((?:spyndynamica|coming-from-spindynamica)[\w-]*\.md)", text):
            assert (DOCS / target).exists(), f"{name} links to {target}"
        assert "coming-from-spindynamica.md" in text or name == "coming-from-spindynamica.md"
