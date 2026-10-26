"""第1回「実装」課題: Gemini APIを使わずに会話履歴の組み立てを確認。"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def ask_module(monkeypatch):
    """実APIアクセスを防ぐためgoogle-genaiの最小限のAPIを差し替える。"""
    google_module = ModuleType("google")
    google_module.__path__ = []
    genai_module = ModuleType("google.genai")
    genai_module.__path__ = []
    errors_module = ModuleType("google.genai.errors")
    types_module = ModuleType("google.genai.types")

    class Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class Part:
        @classmethod
        def from_text(cls, *, text):
            return SimpleNamespace(text=text)

    class APIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    genai_module.Client = object
    genai_module.errors = errors_module
    genai_module.types = types_module
    errors_module.APIError = APIError
    types_module.Content = Content
    types_module.Part = Part
    google_module.genai = genai_module
    config_module = ModuleType("app.config")
    config_module.MODEL = "test-model"
    config_module.api_error_message = lambda code, message: f"APIエラー {code}: {message}"
    config_module.get_api_key = lambda: "dummy-not-a-real-key"
    monkeypatch.setitem(sys.modules, "app.config", config_module)

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    sys.modules.pop("app.ask", None)
    module = importlib.import_module("app.ask")
    yield module
    sys.modules.pop("app.ask", None)


class FakeModels:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append((model, contents))
        return SimpleNamespace(
            text=next(self.answers),
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5
            ),
        )


def test_second_question_contains_previous_question_and_answer(ask_module):
    models = FakeModels(["LoRAは低ランク適応です。", "事前学習済みモデルの更新を効率化します。"])
    client = SimpleNamespace(models=models)
    history = []

    answer1, _ = ask_module.ask_with_history(client, "LoRAとは？", history)
    assert answer1 == "LoRAは低ランク適応です。"
    assert [x.role for x in models.calls[0][1]] == ["user"]

    _, _ = ask_module.ask_with_history(client, "それは何に役立ちますか？", history)
    sent = models.calls[1][1]
    assert [x.role for x in sent] == ["user", "model", "user"]
    assert [x.parts[0].text for x in sent] == [
        "LoRAとは？",
        "LoRAは低ランク適応です。",
        "それは何に役立ちますか？",
    ]
    assert len(history) == 4
    assert len(models.calls) == 2


def test_empty_answer_does_not_pollute_history(ask_module):
    models = FakeModels([""])
    history = []
    with pytest.raises(ValueError, match="テキスト形式"):
        ask_module.ask_with_history(SimpleNamespace(models=models), "質問", history)
    assert history == []


def test_chat_exit_skips_blank_input_and_does_not_call_api_after_exit(
    ask_module, monkeypatch, capsys
):
    models = FakeModels(["テスト回答"])
    client = SimpleNamespace(models=models)
    entries = iter(["", "最初の質問", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(entries))
    ask_module.run_chat(client)
    captured = capsys.readouterr().out
    assert "テスト回答" in captured
    assert "会話を終了しました" in captured
    assert len(models.calls) == 1


def test_usage_metadata_prints_input_and_output(ask_module, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=123, candidates_token_count=45)
    )
    ask_module.print_token_usage(response)
    assert "入力: 123 / 出力: 45" in capsys.readouterr().out


def test_missing_usage_is_explicit(ask_module, capsys):
    ask_module.print_token_usage(SimpleNamespace(usage_metadata=None))
    assert "取得できませんでした" in capsys.readouterr().out
