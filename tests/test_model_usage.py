from yieldmind.model_usage import (
    build_model_usage_record,
    merge_model_usage_records,
    normalize_token_usage,
    summarize_model_usage,
)


def test_normalize_token_usage_accepts_openai_and_responses_names():
    assert normalize_token_usage(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    ) == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    assert normalize_token_usage({"input_tokens": 8, "output_tokens": 3}) == {
        "input_tokens": 8,
        "output_tokens": 3,
        "total_tokens": 11,
    }


def test_usage_summary_distinguishes_partial_provider_reporting():
    metered = build_model_usage_record(
        agent="CandidateAgent",
        stage="candidate",
        purpose="candidate_strategy_generation",
        provider="test",
        model="model-a",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    unmetered = build_model_usage_record(
        agent="OperationAgent",
        stage="operation",
        purpose="model_factory_generation",
        provider="test",
        model="model-a",
    )
    summary = summarize_model_usage([metered, unmetered])

    assert summary["mode"] == "live_partial"
    assert summary["calls_total"] == 2
    assert summary["calls_reported"] == 1
    assert summary["total_tokens"] == 15
    assert summary["per_agent"]["CandidateAgent"]["total_tokens"] == 15
    assert summary["per_agent"]["OperationAgent"]["reported_calls"] == 0


def test_merge_usage_records_deduplicates_call_ids():
    row = build_model_usage_record(
        agent="ModelAgent",
        stage="model",
        purpose="plan",
        provider="test",
        model="model-a",
        usage={"total_tokens": 3},
    )
    merged = merge_model_usage_records([row], [dict(row)])
    assert len(merged) == 1
