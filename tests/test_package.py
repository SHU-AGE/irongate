"""打包脚本测试：排除规则与端到端 zip 结构。"""

from __future__ import annotations

import sys
import unittest
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_package import build, is_excluded  # noqa: E402


class ExclusionRuleTests(unittest.TestCase):
    def test_env_variants(self) -> None:
        self.assertTrue(is_excluded(Path(".env")))
        self.assertTrue(is_excluded(Path("deploy/.env")))
        self.assertTrue(is_excluded(Path(".env.production")))
        self.assertTrue(is_excluded(Path("service.env")))
        # 模板必须保留，否则别人不知道要配什么
        self.assertFalse(is_excluded(Path(".env.example")))

    def test_caches_and_virtualenvs(self) -> None:
        self.assertTrue(is_excluded(Path("irongate/__pycache__/model.cpython-312.pyc")))
        self.assertTrue(is_excluded(Path(".venv/Lib/site-packages/x.py")))
        self.assertTrue(is_excluded(Path("venv/bin/python")))
        self.assertTrue(is_excluded(Path("bench/run.pyc")))
        self.assertTrue(is_excluded(Path(".git/config")))

    def test_normal_files_kept(self) -> None:
        for name in ("README.md", "irongate/guard.py", "artifacts/model.json",
                     "bench/results.json", ".env.example"):
            self.assertFalse(is_excluded(Path(name)), name)


class BuildTests(unittest.TestCase):
    def test_build_creates_zip_with_structure(self) -> None:
        root = ROOT / "dist" / f"_test_{uuid.uuid4().hex}" / "irongate"
        (root / "irongate").mkdir(parents=True)
        (root / "irongate" / "__init__.py").write_text("__version__ = 'x'\n", encoding="utf-8")
        (root / "README.md").write_text("demo\n", encoding="utf-8")
        # 应被排除的三类
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
        (root / ".venv").mkdir()
        (root / ".venv" / "lib.py").write_text("x = 1\n", encoding="utf-8")
        (root / "notes.pyc").write_bytes(b"\x02")

        output = root.parent / "out.zip"
        result = build(root, output, log=lambda *_: None)
        self.assertTrue(output.exists())
        self.assertEqual(result["excluded"], 3)

        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
        self.assertIn("irongate/README.md", names)
        self.assertIn("irongate/irongate/__init__.py", names)
        self.assertIn("irongate/requirements.txt", names)      # 自动补齐
        self.assertIn("irongate/.env.example", names)          # 自动补齐
        self.assertFalse(any("__pycache__" in n for n in names))
        self.assertFalse(any(".venv" in n for n in names))
        self.assertFalse(any(n.endswith(".pyc") for n in names))


if __name__ == "__main__":
    unittest.main()
