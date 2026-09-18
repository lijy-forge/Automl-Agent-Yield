# YieldMind Retrieval Blind Review

只编辑 CSV 中的 `relevance`、`confidence` 和 `notes` 三列。不要查看本地 manifest。

- `2`：候选文本直接回答问题，可单独作为该问题的主要引用。
- `1`：候选文本提供相关背景或部分答案，但不足以单独支撑完整结论。
- `0`：候选文本无关、答非所问，或可能误导回答。
- `confidence`：填写 `high`、`medium` 或 `low`。
- `notes`：可留空；边界模糊、问题本身有歧义或需要多个候选组合时请说明。

逐行只根据 `query` 和 `candidate_text` 判断。不要根据候选编号猜测检索排名，不要修改 `reviewer_id`、`row_id`、`case_id`、`query`、`candidate_code` 或 `candidate_text`。
完成前确认每一行的 `relevance` 和 `confidence` 都已填写。CSV 使用 UTF-8 BOM，可直接用 Excel 打开。
