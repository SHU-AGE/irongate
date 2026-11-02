"""配置层测试：.env 解析、环境变量优先级、CLI config 命令。"""

from __future__ import annotations

import io
import json
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from irongate import cli, config  # noqa: E402


class _TempEnv(unittest.TestCase):
    """在 tests/ 下建一个临时配置文件，用完即删。

    两个刻意的选择：
    * 不用 `tempfile.mkdtemp`：受限环境里"运行时新建的子目录"可能拿不到写权限，
      会导致测试假失败；
    * 文件名不用 `*.env`：部分安全策略禁止创建 .env 文件，而这里测的是解析器。
    """

    def setUp(self) -> None:
        self.path = ROOT / "tests" / f"sample_config_{uuid.uuid4().hex}.txt"

    def tearDown(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


class EnvFileTests(_TempEnv):
    def test_parse_basic(self) -> None:
        self.path.write_text(
            "# 注释\nA=1\nexport B='two words'\nC=\"quoted\"\n\n非法行\nD = 4\n",
            encoding="utf-8",
        )
        values = config.parse_env_file(self.path)
        self.assertEqual(values["A"], "1")
        self.assertEqual(values["B"], "two words")
        self.assertEqual(values["C"], "quoted")
        self.assertEqual(values["D"], "4")
        self.assertNotIn("非法行", values)

    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(config.parse_env_file(Path("no/such/.env")), {})

    def test_load_dotenv_does_not_override(self) -> None:
        self.path.write_text("IRONGATE_PORT=9999\nIRONGATE_HOST=0.0.0.0\n", encoding="utf-8")
        env = {"IRONGATE_PORT": "1234"}
        loaded = config.load_dotenv(self.path, environ=env)
        self.assertEqual(env["IRONGATE_PORT"], "1234")   # 已有值不被覆盖
        self.assertEqual(env["IRONGATE_HOST"], "0.0.0.0")
        self.assertEqual(loaded, {"IRONGATE_HOST": "0.0.0.0"})


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = config.settings_from_env({})
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8100)
        self.assertTrue(settings.fail_closed)
        self.assertEqual(settings.max_input_chars, 40000)
        self.assertIsNone(settings.policy_path)
        self.assertIsNone(settings.canary)
        self.assertTrue(settings.deepseek_enabled)
        self.assertIsNone(settings.deepseek_api_key)
        self.assertEqual(settings.deepseek_model, "deepseek-flash")
        self.assertEqual(settings.deepseek_timeout_seconds, 20)
        self.assertEqual(settings.deepseek_min_confidence, 0.65)

    def test_overrides(self) -> None:
        settings = config.settings_from_env({
            "IRONGATE_HOST": "0.0.0.0",
            "IRONGATE_PORT": "9001",
            "IRONGATE_FAIL_CLOSED": "false",
            "IRONGATE_MAX_INPUT_CHARS": "512",
            "IRONGATE_CANARY": "IG-TEST",
            "IRONGATE_POLICY": "policy.strict.json",
            "DEEPSEEK_API_KEY": "test-secret",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "IRONGATE_DEEPSEEK_TIMEOUT": "12",
            "IRONGATE_DEEPSEEK_MIN_CONFIDENCE": "0.75",
        })
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 9001)
        self.assertFalse(settings.fail_closed)
        self.assertEqual(settings.max_input_chars, 512)
        self.assertEqual(settings.canary, "IG-TEST")
        self.assertTrue(str(settings.policy_path).endswith("policy.strict.json"))
        self.assertEqual(settings.deepseek_api_key, "test-secret")
        self.assertEqual(settings.deepseek_model, "deepseek-v4-pro")
        self.assertEqual(settings.deepseek_timeout_seconds, 12)
        self.assertEqual(settings.deepseek_min_confidence, 0.75)

    def test_bad_numbers_fall_back(self) -> None:
        settings = config.settings_from_env({
            "IRONGATE_PORT": "not-a-port",
            "IRONGATE_MAX_INPUT_CHARS": "-5",
            "IRONGATE_FAIL_CLOSED": "maybe",
        })
        self.assertEqual(settings.port, 8100)
        self.assertEqual(settings.max_input_chars, 256)   # 下限收敛
        self.assertTrue(settings.fail_closed)             # 非法布尔 → 默认

    def test_describe_masks_canary(self) -> None:
        settings = config.settings_from_env({
            "IRONGATE_CANARY": "secret-value",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        })
        self.assertEqual(settings.describe()["canary"], "***")
        self.assertEqual(settings.describe()["deepseek_api_key"], "***")
        self.assertNotIn("deepseek-secret", json.dumps(settings.describe()))


class ConfigCommandTests(unittest.TestCase):
    def test_cli_config_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["config", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("model_path", payload)
        self.assertIn("fail_closed", payload)


if __name__ == "__main__":
    unittest.main()
