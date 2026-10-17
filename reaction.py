"""Generate a best-effort chat reaction without sending chat messages.

Only the Python standard library is required. Configuration is read on every
call, so changes do not require restarting the notification service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import signal
from dataclasses import dataclass, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

logger = logging.getLogger(__name__)
ConfigPath = Union[str, os.PathLike]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class GrokConfig:
    enabled: bool
    cli_path: str
    timeout: float
    max_diff_chars: int
    max_reaction_chars: int
    prompt_template: str
    cli_args: list[str]
    allow_profanity: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.allow_profanity) is not bool:
            raise ValueError("allow_profanity must be a boolean")
        if not isinstance(self.cli_path, str) or not self.cli_path.strip():
            raise ValueError("cli_path must be a nonempty executable name or absolute path")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        for name in ("max_diff_chars", "max_reaction_chars"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.prompt_template, str)
            or "{schedule_diff}" not in self.prompt_template
        ):
            raise ValueError("prompt_template must contain {schedule_diff}")
        if not isinstance(self.cli_args, list) or not all(
            isinstance(arg, str) for arg in self.cli_args
        ):
            raise ValueError("cli_args must be a JSON array of strings")


def _read_json(path: ConfigPath) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Grok configuration must be a JSON object")
    return data


def load_config(config_path: Optional[ConfigPath] = None) -> GrokConfig:
    """Load bundled defaults, optionally overridden by a separate JSON file.

    Priority: explicit path > GROK_REACTION_CONFIG > bundled config.json.
    An override may contain only the settings that need changing.
    Invalid settings raise here; generate_grok_reaction catches these errors.
    """
    data = _read_json(DEFAULT_CONFIG_PATH)
    override_path = config_path if config_path is not None else os.getenv("GROK_REACTION_CONFIG")
    if override_path is not None:
        data.update(_read_json(override_path))
    unknown = data.keys() - {field.name for field in fields(GrokConfig)}
    if unknown:
        raise ValueError("Unknown Grok settings: " + ", ".join(sorted(unknown)))

    if "GROK_REACTIONS_ENABLED" in os.environ:
        raw_enabled = os.environ["GROK_REACTIONS_ENABLED"].strip().lower()
        if raw_enabled in {"1", "true", "yes", "on"}:
            data["enabled"] = True
        elif raw_enabled in {"0", "false", "no", "off"}:
            data["enabled"] = False
        else:
            raise ValueError(
                f"Invalid boolean value for GROK_REACTIONS_ENABLED: {raw_enabled!r}"
            )

    if "GROK_CLI_PATH" in os.environ:
        cli_path = os.environ["GROK_CLI_PATH"].strip()
        if cli_path:
            data["cli_path"] = cli_path

    if "GROK_TIMEOUT_SECONDS" in os.environ:
        try:
            data["timeout"] = float(os.environ["GROK_TIMEOUT_SECONDS"].strip())
        except ValueError:
            raise ValueError("GROK_TIMEOUT_SECONDS must be a valid number") from None

    return GrokConfig(**data)


def _extract_text(data: object) -> Optional[str]:
    if isinstance(data, str):
        return data.strip() or None
    if isinstance(data, dict):
        if data.get("error") or data.get("is_error"):
            return None
        for key in ("response", "text", "result"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(data, list):
        parts = [_extract_text(item) for item in data]
        return "\n".join(part for part in parts if part) or None
    return None


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Kill and reap; also stop the dedicated process group on Linux/macOS."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    except OSError:
        logger.warning("Could not terminate Grok process", exc_info=True)
    try:
        await asyncio.wait_for(process.communicate(), timeout=5)
    except (asyncio.TimeoutError, OSError):
        logger.warning("Grok process cleanup did not finish within 5 seconds")


async def generate_grok_reaction(
    schedule_diff: str,
    cli_path: Optional[str] = None,
    timeout: Optional[float] = None,
    *,
    enabled: Optional[bool] = None,
    config_path: Optional[ConfigPath] = None,
) -> Optional[str]:
    """Return a reaction or None; configuration/CLI errors never break mailing.

    The original positional cli_path and timeout arguments remain supported.
    The enabled parameter allows explicit control from external settings.
    Task cancellation stops the subprocess and propagates to the caller.
    """
    process = None
    try:
        config = load_config(config_path)
        if enabled is not None:
            config = replace(config, enabled=enabled)
        if cli_path is not None:
            config = replace(config, cli_path=cli_path)
        if timeout is not None:
            config = replace(config, timeout=timeout)
        if not config.enabled or not schedule_diff.strip():
            return None

        diff = schedule_diff[:config.max_diff_chars]
        profanity_instruction = (
            "Используй богатую и разнообразную разговорную речь с неформальным матом "
            "(бля / блять, заебись, охуенно, пиздец, ебать, ебануться, нихуя себе, съёбывать, похуй, кайф). "
            "Категорически не зацикливайся на одном слове и не лепи 'нахуй' в конец каждого предложения как паразит: "
            "мат должен быть разнообразным и органично вписываться в живую мысль."
            if config.allow_profanity
            else "Пиши без мата и его замаскированных вариантов."
        )
        prompt = config.prompt_template.replace("{schedule_diff}", diff)
        prompt += "\n\n" + profanity_instruction
        resolved_cli = shutil.which(config.cli_path) or config.cli_path
        cmd = [resolved_cli, *config.cli_args, "-p", prompt]

        # Do not expose the notification project's working directory to Grok.
        # Authentication and user-level Grok settings still come from its HOME.
        with TemporaryDirectory(prefix="schedule-grok-") as workdir:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=workdir,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=(os.name == "posix"),
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=config.timeout
                )
            except BaseException:
                if process is not None:
                    await _stop_process(process)
                raise

        if process.returncode != 0:
            logger.warning(
                "Grok CLI exited with code %s: %s",
                process.returncode,
                stderr.decode("utf-8", errors="replace").strip()[:500],
            )
            return None

        raw = stdout.decode("utf-8-sig", errors="replace").strip()
        try:
            reaction = _extract_text(json.loads(raw))
        except json.JSONDecodeError:
            reaction = raw or None
        if reaction:
            cleaned = reaction.strip()
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                cleaned.startswith('«') and cleaned.endswith('»')
            ):
                cleaned = cleaned[1:-1].strip()
            return cleaned[:config.max_reaction_chars].strip() or None
        return None
    except asyncio.TimeoutError:
        logger.warning("Grok CLI generation timed out after %ss", config.timeout)
    except FileNotFoundError:
        logger.warning("Grok executable or configuration file was not found", exc_info=True)
    except Exception:
        logger.exception("Could not generate Grok reaction")
    return None
