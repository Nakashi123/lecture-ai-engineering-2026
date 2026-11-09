"""第3回演習：確認問題ツールが用語集を参照することを確かめる。"""

import json
from app import tools


def test_make_quiz_from_glossary():
    result = json.loads(tools.make_quiz(2))
    assert result["lecture"] == 2
    assert result["question"].endswith("とは何ですか？")
    assert result["answer"]


def test_make_quiz_unknown_lecture():
    assert "ありません" in tools.make_quiz(999)
    assert "1以上" in tools.make_quiz(0)
