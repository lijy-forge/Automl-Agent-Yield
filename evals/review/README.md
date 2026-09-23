# YieldMind Retrieval Blind Review

只编辑 CSV 中的 `relevance`、`confidence` 和 `notes` 三列。不要查看本地 manifest。
`query` / `candidate_text` 是英文原文，`query_zh` / `candidate_text_zh` 是机器生成的中文辅助翻译；有冲突时以英文原文为准。

- `2`：候选文本直接回答问题，可单独作为该问题的主要引用。
- `1`：候选文本提供相关背景或部分答案，但不足以单独支撑完整结论。
- `0`：候选文本无关、答非所问，或可能误导回答。
- `confidence`：填写 `high`、`medium` 或 `low`。
- `notes`：可留空；边界模糊、问题本身有歧义或需要多个候选组合时请说明。

逐行根据 `query`、英文原文和中文辅助翻译判断。不要根据候选编号猜测检索排名，不要修改 `reviewer_id`、`row_id`、`case_id`、`query`、`query_zh`、`candidate_code`、英文或中文文本。
完成前确认每一行的 `relevance` 和 `confidence` 都已填写。CSV 使用 UTF-8 BOM，可直接用 Excel 打开。

## 双人标注分析与裁决

双人标注完成后，运行 `scripts/analyze_yieldmind_retrieval_reviews.py` 校验候选文本哈希、不可编辑字段和标签完整性，并生成一致性报告与裁决表。分析脚本不会改写原始 CSV。

裁决时只编辑 `yieldmind_retrieval_review_adjudication.csv` 的 `adjudicated_relevance` 和 `adjudication_notes`：

- `adjudicated_relevance` 仍使用 `0`、`1`、`2`，需要根据原问题、候选英文原文和标注规则重新判断。
- `adjudication_notes` 简短说明采用该等级的依据；不要通过平均、四舍五入或按置信度自动决定。
- 两位标注一致的行无需再次填写；两份原始标注文件均保持不变。

裁决完成后，运行 `scripts/finalize_yieldmind_retrieval_review.py`。脚本会重新计算原始分歧并拒绝保护字段改动、漏填和非法标签，随后生成完整 Gold Label CSV、最终指标和 Top-1 Bad Case。最终报告明确记录所有输入文件哈希以及 `real_llm_calls=0`。
