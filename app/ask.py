"""第1回: 素のLLMに質問するスクリプト。会話モードにも対応。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"  # 従来の1回質問
    uv run python -m app.ask                    # 講座に固有の質問
    uv run python -m app.ask --chat             # 会話履歴を保持して連続質問

--chat では質問と回答を順に履歴に追加し、次のAPI呼び出しにも渡す。
「終了」または「exit」を入力すると終了する。
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_token_usage(response: object) -> None:
    """APIの返すusage_metadataから入出力トークン数を表示する。"""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        print("[トークン数] 取得できませんでした")
        return

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    input_display = input_tokens if input_tokens is not None else "不明"
    output_display = output_tokens if output_tokens is not None else "不明"
    print(f"[トークン数] 入力: {input_display} / 出力: {output_display}")


def ask(question: str) -> str:
    """従来どおり、独立した質問を1回だけ行う。"""
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_token_usage(response)
    return response.text or "（テキスト形式の回答はありませんでした）"


def ask_with_history(
    client: genai.Client, question: str, history: list[types.Content]
) -> tuple[str, object]:
    """履歴を渡して質問し、成功した質問・回答のみ履歴に追加する。

    historyは同じリストを使い回す。APIエラーや空回答があった場合、
    不完全なやり取りは履歴に追加しない。
    """
    user_turn = types.Content(
        role="user", parts=[types.Part.from_text(text=question)]
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[*history, user_turn],
    )
    answer = response.text
    if not answer:
        raise ValueError("モデルからテキスト形式の回答が返りませんでした。")

    model_turn = types.Content(
        role="model", parts=[types.Part.from_text(text=answer)]
    )
    history.extend((user_turn, model_turn))
    return answer, response


def run_chat(client: genai.Client) -> None:
    """同一セッション内で連続質問を受け付ける。"""
    history: list[types.Content] = []
    print("[会話モード] 質問を入力してください。終了するには「終了」または「exit」。")

    while True:
        try:
            question = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会話を終了しました。")
            break

        if question.lower() in {"exit", "quit", "終了"}:
            print("会話を終了しました。")
            break
        if not question:
            continue

        try:
            answer, response = ask_with_history(client, question, history)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue
        except ValueError as e:
            print(f"[応答エラー] {e}")
            continue

        print(f"[回答] {answer}")
        print_token_usage(response)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する（--chatで会話モード）")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って連続質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        client = genai.Client(api_key=get_api_key())
        run_chat(client)
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
