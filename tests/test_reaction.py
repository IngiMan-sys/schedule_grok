import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from schedule_grok import generate_grok_reaction, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "grok.json"
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("GROK_REACTION_CONFIG", None)

    def write_config(self, value):
        self.path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return self.path

    def test_bundled_defaults_include_requested_restrictions(self):
        config = load_config()
        self.assertEqual(config.timeout, 120)
        self.assertIn("--output-format", config.cli_args)
        self.assertIn("json", config.cli_args)
        self.assertIn("--no-plan", config.cli_args)
        self.assertIn("--no-subagents", config.cli_args)

    def test_partial_override_preserves_defaults(self):
        config = load_config(self.write_config({"timeout": 12, "enabled": False}))
        self.assertEqual(config.timeout, 12)
        self.assertFalse(config.enabled)
        self.assertEqual(config.cli_args, load_config().cli_args)

    def test_explicit_path_has_priority_over_environment(self):
        os.environ["GROK_REACTION_CONFIG"] = str(self.path.with_name("missing.json"))
        config = load_config(self.write_config({"timeout": 7}))
        self.assertEqual(config.timeout, 7)

    def test_environment_override(self):
        os.environ["GROK_REACTION_CONFIG"] = str(self.write_config({"timeout": 4}))
        self.assertEqual(load_config().timeout, 4)

    def test_grok_direct_env_vars(self):
        os.environ["GROK_REACTIONS_ENABLED"] = "false"
        os.environ["GROK_CLI_PATH"] = "custom-grok"
        os.environ["GROK_TIMEOUT_SECONDS"] = "45"
        config = load_config()
        self.assertFalse(config.enabled)
        self.assertEqual(config.cli_path, "custom-grok")
        self.assertEqual(config.timeout, 45.0)

        os.environ["GROK_REACTIONS_ENABLED"] = "true"
        self.assertTrue(load_config().enabled)

    def test_grok_direct_env_vars_invalid(self):
        os.environ["GROK_REACTIONS_ENABLED"] = "not-a-bool"
        with self.assertRaises(ValueError):
            load_config()

        os.environ.pop("GROK_REACTIONS_ENABLED")
        os.environ["GROK_TIMEOUT_SECONDS"] = "invalid-num"
        with self.assertRaises(ValueError):
            load_config()

    def test_rejects_invalid_values(self):
        for data in (
            {"enabled": "false"}, {"allow_profanity": "false"},
            {"timeout": 0}, {"timeout": True},
            {"timeout": float("nan")}, {"timeout": float("inf")},
            {"cli_path": ""}, {"cli_args": "--flag"}, {"cli_args": [42]},
            {"max_diff_chars": 0}, {"max_reaction_chars": True},
            {"prompt_template": "missing placeholder"}, {"timeuot": 10}, [],
        ):
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                load_config(self.write_config(data))


class ReactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "grok.json"
        self.write_config({})
        self.process = AsyncMock()
        self.process.returncode = 0
        self.process.communicate.return_value = (b'{"text": "reaction"}', b"")
        self.spawn = AsyncMock(return_value=self.process)
        self.patcher = patch("schedule_grok.reaction.asyncio.create_subprocess_exec", self.spawn)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write_config(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def generate(self, diff="Математика отменена", **kwargs):
        return await generate_grok_reaction(diff, config_path=self.path, **kwargs)

    async def test_plain_and_json_response_formats(self):
        for raw, expected in (
            ("  Отлично, снова перемены.  ", "Отлично, снова перемены."),
            ('{"response": "response"}', "response"),
            ('{"text": "text"}', "text"),
            ('{"result": "result"}', "result"),
            ('{"response": null, "text": "fallback"}', "fallback"),
            ('[{"text": "first"}, {"text": "second"}]', "first\nsecond"),
            ('"a JSON string"', "a JSON string"),
            ('{"error": "auth failed", "text": "diagnostic"}', None),
            ('{"is_error": true, "text": "diagnostic"}', None),
            ('{"text": 123}', None), ('{"unknown": "data"}', None),
            ("[]", None), ("null", None), ("123", None), ("", None), ("  ", None),
        ):
            with self.subTest(raw=raw):
                self.process.communicate.return_value = (raw.encode(), b"")
                self.assertEqual(await self.generate(), expected)

    async def test_utf8_bom_and_invalid_stderr_bytes(self):
        self.process.communicate.return_value = (b'\xef\xbb\xbf{"text":"ok"}', b"\xff")
        self.assertEqual(await self.generate(), "ok")
        self.process.returncode = 1
        with self.assertLogs("schedule_grok.reaction", level="WARNING"):
            self.assertIsNone(await self.generate())

    async def test_disabled_and_empty_diff_skip_cli(self):
        self.write_config({"enabled": False})
        self.assertIsNone(await self.generate())
        self.write_config({})
        self.assertIsNone(await self.generate(" \n"))
        self.assertIsNone(await self.generate("test", enabled=False))
        self.spawn.assert_not_called()

    async def test_config_reloads_for_each_call(self):
        self.assertEqual(await self.generate(), "reaction")
        self.write_config({"enabled": False})
        self.assertIsNone(await self.generate())
        self.spawn.assert_awaited_once()

    async def test_prompt_is_single_argument_and_braces_are_literal(self):
        diff = '$(do-not-run) `test` {text} "quoted"\nnew line'
        self.write_config({"prompt_template": "Diff: {schedule_diff}"})
        await self.generate(diff, cli_path="/opt/grok/bin/grok", timeout=3)
        args, kwargs = self.spawn.call_args
        self.assertEqual(args[0], "/opt/grok/bin/grok")
        self.assertEqual(args[-2], "-p")
        self.assertTrue(args[-1].startswith("Diff: " + diff + "\n\n"))
        self.assertEqual(kwargs["stdin"], asyncio.subprocess.DEVNULL)
        self.assertNotEqual(kwargs["cwd"], os.getcwd())
        self.assertFalse(Path(kwargs["cwd"]).exists())
        self.assertEqual(kwargs["start_new_session"], os.name == "posix")

    async def test_input_and_output_length_limits(self):
        self.write_config({"max_diff_chars": 3, "max_reaction_chars": 4})
        self.assertEqual(await self.generate("abcdef"), "reac")
        prompt = self.spawn.call_args.args[-1]
        self.assertIn("abc", prompt)
        self.assertNotIn("abcdef", prompt)

    async def test_nonzero_exit_is_nonfatal(self):
        self.process.returncode = 2
        self.process.communicate.return_value = (b"", b"unknown flag")
        with self.assertLogs("schedule_grok.reaction", level="WARNING") as logs:
            self.assertIsNone(await self.generate())
        self.assertIn("unknown flag", "\n".join(logs.output))

    async def test_missing_cli_is_nonfatal(self):
        self.spawn.side_effect = FileNotFoundError("missing grok")
        with self.assertLogs("schedule_grok.reaction", level="WARNING"):
            self.assertIsNone(await self.generate())

    async def test_bad_or_missing_config_is_nonfatal(self):
        for raw in ('{"timeout": false}', '{"enabled":'):
            self.path.write_text(raw, encoding="utf-8")
            with self.assertLogs("schedule_grok.reaction", level="ERROR"):
                self.assertIsNone(await self.generate())
        self.path.unlink()
        with self.assertLogs("schedule_grok.reaction", level="WARNING"):
            self.assertIsNone(await self.generate())
        self.spawn.assert_not_called()

    async def test_timeout_stops_and_reaps_process(self):
        self.process.communicate.side_effect = [asyncio.TimeoutError(), (b"", b"")]
        with patch("schedule_grok.reaction.os.killpg", create=True) as killpg:
            with self.assertLogs("schedule_grok.reaction", level="WARNING"):
                self.assertIsNone(await self.generate())
            if os.name == "posix":
                killpg.assert_called_once()
        self.assertEqual(self.process.communicate.await_count, 2)

    async def test_cancellation_cleans_up_and_propagates(self):
        self.process.communicate.side_effect = [asyncio.CancelledError(), (b"", b"")]
        with patch("schedule_grok.reaction._stop_process", new_callable=AsyncMock) as stop:
            with self.assertRaises(asyncio.CancelledError):
                await self.generate()
            stop.assert_awaited_once_with(self.process)


class SubprocessIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise actual pipe I/O and process cleanup without a Grok installation."""

    async def test_real_subprocess_stdout_and_closed_stdin(self):
        code = (
            "import json, sys; "
            "assert sys.stdin.read() == ''; "
            "print(json.dumps({'text': sys.argv[-1]}))"
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "cli_path": sys.executable,
                "cli_args": ["-c", code],
                "prompt_template": "Reaction: {schedule_diff}",
            }), encoding="utf-8")
            result = await generate_grok_reaction("test", config_path=config_path)
        self.assertTrue(result.startswith("Reaction: test\n\n"))

    async def test_real_timeout_reaps_process(self):
        import schedule_grok.reaction as module

        original_spawn = asyncio.create_subprocess_exec
        spawned = []

        async def capture_spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            spawned.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "cli_path": sys.executable,
                "cli_args": ["-c", "import time; time.sleep(60)"],
                "timeout": 0.15,
            }), encoding="utf-8")
            with patch.object(module.asyncio, "create_subprocess_exec", capture_spawn):
                with self.assertLogs("schedule_grok.reaction", level="WARNING"):
                    result = await generate_grok_reaction("test", config_path=config_path)
        self.assertIsNone(result)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].returncode)


if __name__ == "__main__":
    unittest.main()
