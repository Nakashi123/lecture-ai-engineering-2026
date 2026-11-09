"""第3回実装：保存ツールの承認／拒否と、保存先の安全性を確認する。"""

from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent


def _pending_save():
    model = TestModel(call_tools=["save_learning_log"])
    result = agent.run_sync("学習記録を保存して", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_name == "save_learning_log"
    return result


def _resume(result, approved: bool):
    call = result.output.approvals[0]
    return agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: approved}),
        model=TestModel(call_tools=[]),
    )


def test_save_requires_approval_and_does_not_write_early(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)
    result = _pending_save()
    assert not log_dir.exists()  # 承認待ちではファイルが生成されない
    _resume(result, True)
    saved = list(log_dir.glob("*.md"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8").startswith("# ")


def test_denial_does_not_save(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)
    result = _pending_save()
    _resume(result, False)
    assert not log_dir.exists()


def test_unique_filenames_and_empty_record(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)
    assert "保存していません" in tools.save_learning_log("", "内容")
    assert not log_dir.exists()
    tools.save_learning_log("LoRA", "要点A")
    tools.save_learning_log("LoRA", "要点B")
    assert len(list(log_dir.glob("*.md"))) == 2
