# 第2回・基礎課題：エポック数による比較（実行計画）

## 目的
配布された LoRA ノートブックの `CONFIG["num_train_epochs"]` だけを20から10へ変え、同じ train/valid/test と生成設定の下で、形式遵守率・内容正解率・最終ログ損失・学習時間を比較する。

## 実行する場所
Google Colab の T4 GPU。講座配布ノートブックをコピーした `notebooks/day2_basic_epochs_compare.ipynb` を使用する。その他の学習条件は変更しない。

## 実行順序（実行前に各ステップを確認）

1. ノートブックをColabで開き、「ファイル → ドライブにコピーを保存」。ランタイムをT4 GPUにする。
2. `num_train_epochs=20` のままセクション0〜7を上から実行する。セクション8（4bit）は任意、比較実験には使わない。
3. `/content/qlora_lab/result.json` をダウンロードし、PCで `result_20epochs.json` に名前を変えて保存する。必要に応じて `eval_before.json` / `eval_after.json` も保存する。
4. **ランタイム → セッションを再起動**。ノートブックの `CONFIG["num_train_epochs"]` だけを10に変更する。
5. 再びセクション0〜7を実行する。結果を `result_10epochs.json` として保存する。
6. 2つのJSONの `config` を確認し、学習設定はエポック数以外一致することを確かめる。
7. `before.format_rate`, `before.content_rate`, `after.format_rate`, `after.content_rate`, `training_metrics.final_logged_loss`, `training_metrics.train_runtime_seconds` を比較し、失敗した質問の出力も目視する。
8. `lectures/02/answer.md` の演習欄・宿題欄に **自分で確認した数値** と結果の解釈を記入し、自分のforkの`lec02`→`main`でPRを作成する。
9. QuriのQ1=基礎、Q2=公開リポジトリURL、Q3=400文字以内の振り返りを提出する。

## 記録項目と注意点

- `format_rate` は指定2キー・型を含む形式遵守率。
- `content_rate` はノートブック上では必要語句と`needs_review`で判定され、形式違反の出力でもJSON部分を読み取れたときは内容を別に採点する。形式・内容の両方の正解は`both_rate`。
- 最終lossは`training_metrics.final_logged_loss`（最後のログ区間の学習損失）、全学習の平均`mean_train_loss`とは区別する。
- 学習時間は`train_runtime_seconds`。評価に要した`before.seconds`/`after.seconds`は学習時間ではない。
- GPUはseedを固定しても完全には再現しない。成績が上がった・下がった理由を断定しない。
- 第1回の`lec01`PRには手を付けない。講座本体リポジトリへPRしない。
- ここに記載された手順は実験計画であり、学習を実行したという記録ではない。
