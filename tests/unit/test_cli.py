"""Unit tests for the CLI argument parsing and main() orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localtalk.cli import main, parse_args
from localtalk.models.config import AppConfig

pytestmark = pytest.mark.unit


# ────────────────────────── parse_args ──────────────────────────


class TestParseArgs:
    def test_default_args(self):
        with patch("sys.argv", ["localtalk"]):
            args = parse_args()
        assert args.model == "mlx-community/gpt-oss-20b-MXFP4-Q8"
        assert args.llm_provider == "mlx"
        assert args.whisper_model == "turbo"
        assert args.temperature == 0.7
        assert args.top_p == 1.0
        assert args.max_tokens == 512
        assert args.reasoning == "low"
        assert args.no_tts is False
        assert args.stats is False
        assert args.test_mic is False
        assert args.vad_mode == "auto"
        assert args.vad_threshold == 0.5
        assert args.vad_min_speech_ms == 250
        assert args.system_prompt is None
        assert args.system_prompt_file is None
        assert args.enable_web is False
        assert args.no_web is False
        assert args.skip_network_probe is False

    @pytest.mark.parametrize(
        "flag",
        ["--no-tts", "--stats", "--test-mic", "--enable-web", "--skip-network-probe"],
    )
    def test_boolean_flags(self, flag):
        with patch("sys.argv", ["localtalk", flag]):
            args = parse_args()
        attr = flag.lstrip("-").replace("-", "_")
        assert getattr(args, attr) is True

    @pytest.mark.parametrize(
        ("mode"),
        ["auto", "manual", "off"],
    )
    def test_vad_mode_choices(self, mode):
        with patch("sys.argv", ["localtalk", "--vad-mode", mode]):
            args = parse_args()
        assert args.vad_mode == mode

    def test_invalid_vad_mode_exits(self):
        with patch("sys.argv", ["localtalk", "--vad-mode", "invalid"]), pytest.raises(SystemExit):
            parse_args()

    @pytest.mark.parametrize(
        ("whisper_model"),
        ["tiny", "base.en", "turbo", "large-v3"],
    )
    def test_whisper_model_choices(self, whisper_model):
        with patch("sys.argv", ["localtalk", "--whisper-model", whisper_model]):
            args = parse_args()
        assert args.whisper_model == whisper_model

    def test_invalid_whisper_model_exits(self):
        with patch("sys.argv", ["localtalk", "--whisper-model", "huge"]), pytest.raises(SystemExit):
            parse_args()

    def test_custom_model(self):
        with patch("sys.argv", ["localtalk", "--model", "custom/model"]):
            args = parse_args()
        assert args.model == "custom/model"

    @pytest.mark.parametrize(
        ("level"),
        ["low", "medium", "high"],
    )
    def test_reasoning_choices(self, level):
        with patch("sys.argv", ["localtalk", "--reasoning", level]):
            args = parse_args()
        assert args.reasoning == level

    def test_invalid_reasoning_exits(self):
        with patch("sys.argv", ["localtalk", "--reasoning", "maximum"]), pytest.raises(SystemExit):
            parse_args()

    def test_show_reasoning_flag(self):
        with patch("sys.argv", ["localtalk", "--show-reasoning"]):
            args = parse_args()
        assert args.show_reasoning is True

    def test_show_reasoning_default_false(self):
        with patch("sys.argv", ["localtalk"]):
            args = parse_args()
        assert args.show_reasoning is False

    def test_numeric_args(self):
        with patch("sys.argv", ["localtalk", "--temperature", "0.3", "--max-tokens", "500", "--top-p", "0.9"]):
            args = parse_args()
        assert args.temperature == 0.3
        assert args.max_tokens == 500
        assert args.top_p == 0.9

    def test_system_prompt_inline(self):
        with patch("sys.argv", ["localtalk", "--system-prompt", "Be brief"]):
            args = parse_args()
        assert args.system_prompt == "Be brief"

    def test_system_prompt_file_flag(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom prompt from file.")
        with patch("sys.argv", ["localtalk", "--system-prompt-file", str(prompt_file)]):
            args = parse_args()
        assert args.system_prompt_file == str(prompt_file)


# ────────────────────────── main() ──────────────────────────


class TestMain:
    def _run_main_with_mocks(self, argv: list[str], tmp_path: Path | None = None) -> MagicMock:
        """Run main() with VoiceAssistant and AudioService mocked. Returns the VoiceAssistant mock."""
        mock_assistant = MagicMock()

        with (
            patch("sys.argv", argv),
            patch("localtalk.core.assistant.VoiceAssistant", return_value=mock_assistant) as mock_va_class,
        ):
            main()
            return mock_va_class, mock_assistant

    def test_main_creates_and_runs_assistant(self):
        mock_va_class, mock_assistant = self._run_main_with_mocks(["localtalk"])
        assert mock_va_class.called
        assert mock_assistant.run.called

    def test_main_maps_model_args_to_config(self):
        mock_va_class, _ = self._run_main_with_mocks(
            ["localtalk", "--model", "custom/model", "--temperature", "0.3", "--max-tokens", "500"],
        )
        config: AppConfig = mock_va_class.call_args[0][0]
        assert config.mlx_lm.model == "custom/model"
        assert config.mlx_lm.temperature == 0.3
        assert config.mlx_lm.max_tokens == 500

    def test_main_maps_whisper_model(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--whisper-model", "tiny"])
        config = mock_va_class.call_args[0][0]
        assert config.whisper.model_size == "tiny"

    def test_main_maps_reasoning_effort(self):
        from localtalk.models.config import ReasoningLevel

        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--reasoning", "high"])
        config = mock_va_class.call_args[0][0]
        assert config.mlx_lm.reasoning_effort == ReasoningLevel.HIGH

    def test_main_reasoning_defaults_to_low(self):
        from localtalk.models.config import ReasoningLevel

        mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
        config = mock_va_class.call_args[0][0]
        assert config.mlx_lm.reasoning_effort == ReasoningLevel.LOW

    def test_main_no_tts_sets_backend_none(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--no-tts"])
        config = mock_va_class.call_args[0][0]
        assert config.tts_backend == "none"

    def test_main_tts_enabled_by_default(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
        config = mock_va_class.call_args[0][0]
        assert config.tts_backend == "chatterbox"
        assert config.audio.save_generated_audio is False

    def test_main_save_audio_enables_generated_audio_files(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--save-audio"])
        config = mock_va_class.call_args[0][0]
        assert config.audio.save_generated_audio is True

    def test_main_stats_flag(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--stats"])
        config = mock_va_class.call_args[0][0]
        assert config.show_stats is True

    def test_main_vad_auto(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--vad-mode", "auto"])
        config = mock_va_class.call_args[0][0]
        assert config.audio.use_vad is True
        assert config.audio.vad_auto_start is True

    def test_main_vad_manual(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--vad-mode", "manual"])
        config = mock_va_class.call_args[0][0]
        assert config.audio.use_vad is True
        assert config.audio.vad_auto_start is False

    def test_main_vad_off(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--vad-mode", "off"])
        config = mock_va_class.call_args[0][0]
        assert config.audio.use_vad is False
        assert config.audio.vad_auto_start is False

    def test_main_vad_threshold_and_timing(self):
        mock_va_class, _ = self._run_main_with_mocks(
            ["localtalk", "--vad-threshold", "0.7", "--vad-min-speech-ms", "500"],
        )
        config = mock_va_class.call_args[0][0]
        assert config.audio.vad_threshold == 0.7
        assert config.audio.vad_min_speech_duration_ms == 500

    def test_main_enable_web_sets_policy_on(self, monkeypatch):
        monkeypatch.delenv("LOCALTALK_ENABLE_WEB", raising=False)
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--enable-web"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "on"

    def test_main_no_web_sets_policy_off(self, monkeypatch):
        monkeypatch.delenv("LOCALTALK_ENABLE_WEB", raising=False)
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--no-web"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "off"

    def test_main_no_web_wins_over_env_enable(self, monkeypatch):
        monkeypatch.setenv("LOCALTALK_ENABLE_WEB", "1")
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--no-web"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "off"

    def test_main_env_enable_web_when_no_flag(self, monkeypatch):
        monkeypatch.setenv("LOCALTALK_ENABLE_WEB", "1")
        mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "on"

    def test_main_env_disable_web_when_no_flag(self, monkeypatch):
        monkeypatch.setenv("LOCALTALK_ENABLE_WEB", "0")
        mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "off"

    def test_main_default_web_policy_auto(self, monkeypatch):
        monkeypatch.delenv("LOCALTALK_ENABLE_WEB", raising=False)
        mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.policy == "auto"

    def test_main_skip_network_probe(self):
        mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--skip-network-probe"])
        config = mock_va_class.call_args[0][0]
        assert config.web_tools.startup_probe is False

    def test_main_system_prompt_file_overrides_inline(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Prompt from file.")
        mock_va_class, _ = self._run_main_with_mocks(
            ["localtalk", "--system-prompt", "inline prompt", "--system-prompt-file", str(prompt_file)],
        )
        config = mock_va_class.call_args[0][0]
        assert config.system_prompt == "Prompt from file."

    def test_main_system_prompt_inline_when_no_file(self, tmp_path, monkeypatch):
        """--system-prompt is only used when no prompt file is available at all.

        The default prompts/default.txt takes precedence over --system-prompt.
        We temporarily move the default file so the inline prompt gets used.
        """
        default_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "default.txt"
        backup = default_path.read_text() if default_path.exists() else None
        try:
            default_path.unlink(missing_ok=True)
            mock_va_class, _ = self._run_main_with_mocks(["localtalk", "--system-prompt", "Be concise"])
            config = mock_va_class.call_args[0][0]
            assert config.system_prompt == "Be concise"
        finally:
            if backup is not None:
                default_path.write_text(backup)

    def test_main_missing_prompt_file_returns_early(self):
        """A missing --system-prompt-file should print an error and return without creating the assistant."""
        with (
            patch("sys.argv", ["localtalk", "--system-prompt-file", "/nonexistent/prompt.txt"]),
            patch("localtalk.core.assistant.VoiceAssistant") as mock_va,
        ):
            main()
        assert not mock_va.called

    def test_main_test_mic_exits_without_assistant(self):
        """--test-mic should not create a VoiceAssistant."""
        with (
            patch("sys.argv", ["localtalk", "--test-mic"]),
            patch("localtalk.core.assistant.VoiceAssistant") as mock_va,
            patch("localtalk.services.audio.AudioService") as mock_audio_cls,
        ):
            mock_audio = MagicMock()
            mock_audio.test_microphone.return_value = True
            mock_audio_cls.return_value = mock_audio
            main()
        assert not mock_va.called
        assert mock_audio.test_microphone.called

    def test_main_interrupt_during_startup_prints_clean_goodbye(self, capsys):
        with patch("localtalk.cli._main", side_effect=KeyboardInterrupt):
            main()

        assert capsys.readouterr().out == "\nGoodbye.\n"

    def test_main_default_prompt_file_loaded(self, tmp_path, monkeypatch):
        """When no --system-prompt or --system-prompt-file, default prompts/default.txt is loaded."""
        default_prompt = "Default prompt from file."
        # The code looks for prompts/default.txt relative to cli.py's location
        # which is src/localtalk/cli.py → parent.parent.parent / prompts / default.txt
        prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
        original_content = None
        default_path = prompts_dir / "default.txt"
        if default_path.exists():
            original_content = default_path.read_text()

        try:
            default_path.write_text(default_prompt)
            mock_va_class, _ = self._run_main_with_mocks(["localtalk"])
            config = mock_va_class.call_args[0][0]
            assert config.system_prompt == default_prompt
        finally:
            if original_content is not None:
                default_path.write_text(original_content)
