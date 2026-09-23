const AGENTS = [
  { name: "Agent Manager", role: "领域编排、审批、路由与修订", icon: "🧠" },
  { name: "DataAgent", role: "数据校验、画像与运行上下文", icon: "📊" },
  { name: "SearchAgent", role: "外部证据与领域知识检索", icon: "🔎" },
  { name: "CandidateAgent", role: "生成模型、机理与组合候选", icon: "🧩" },
  { name: "ModelAgent", role: "生成可执行建模方案", icon: "⚙️" },
  { name: "OperationAgent", role: "生成代码、执行、修错与重试", icon: "💻" },
];
const STAGE_LABELS = {
  start: "初始化运行", prepare_data: "准备数据", profile: "数据画像", retrieve_evidence: "检索证据",
  evaluate: "评估固定基线", candidate_benchmark: "评测候选策略", verify_artifacts: "验收产物",
  report: "生成报告", local_repair: "局部修复", replan: "重新规划", cancel: "取消运行", finish: "完成运行",
  prepare: "准备领域工作区", data: "领域数据处理", requirements: "需求分析", search: "外部证据搜索",
  candidate: "候选策略生成", model: "模型方案生成", pre_execution: "执行前审批",
  operation: "执行建模方案", review: "结果复核", revision_feedback: "生成修订反馈",
};
const STATUS_LABELS = { pending: "等待", running: "执行中", passed: "已完成", failed: "失败", cancelled: "已取消", skipped: "未参与" };
const FINAL_SELECTION_REASONS = {
  fixed_baseline_selected_lower_oof_rmse: "固定基线在相同交叉验证折上的 OOF RMSE 更低。",
  fixed_baseline_selected_tie: "两者 OOF RMSE 持平，按规则优先选择更简单的固定基线。",
  candidate_selected_lower_oof_rmse: "候选策略在相同交叉验证折上的 OOF RMSE 更低。",
  not_comparable: "候选策略与固定基线缺少同协议可比结果，未产生最终推荐。",
  legacy_candidate_selection: "历史运行未保存全局选择结果。",
};
const state = {
  currentRunId: "", currentTaskId: "", taskPollTimer: null, result: null, chartPoints: [],
  agentStates: {}, liveEvents: [], liveStages: [], behaviorEvents: [], sessionId: "",
};
const SESSION_STORAGE_KEY = "yieldmind-session-id-v1";
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function formatNumber(value, digits = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "--";
}
function formatTime(value) {
  const time = Number(value);
  return Number.isFinite(time) ? new Date(time * 1000).toLocaleString("zh-CN", { hour12: false }) : "--";
}
function toast(message, error = false) {
  const box = $("toast");
  box.textContent = message;
  box.className = `toast visible${error ? " error" : ""}`;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { box.className = "toast"; }, 3600);
}
async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = { detail: `HTTP ${response.status}` }; }
  if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

async function ensureSessionTurn(content) {
  let sessionId = state.sessionId || window.localStorage.getItem(SESSION_STORAGE_KEY) || "";
  if (sessionId) {
    try {
      await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
    } catch (_error) {
      sessionId = "";
    }
  }
  if (!sessionId) {
    const created = await requestJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_id: "yieldmind_web" }),
    });
    sessionId = created.session.session_id;
    window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  state.sessionId = sessionId;
  const message = await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      content,
      idempotency_key: `web-turn-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      max_context_tokens: 1200,
      create_run_if_requested: false,
    }),
  });
  return { sessionId, turnId: message.turn.turn_id };
}
function setStatus(id, ok, label) {
  const item = $(id);
  item.classList.toggle("ok", ok === true);
  item.classList.toggle("bad", ok === false);
  if (label) item.lastChild.textContent = label;
}

function setLiveState(status, label) {
  const box = $("agent-live-state");
  box.dataset.state = status;
  box.querySelector("span").textContent = label || STATUS_LABELS[status] || status;
}
function resetAgentActivity() {
  state.agentStates = Object.fromEntries(AGENTS.map((agent) => [agent.name, {
    status: "pending", stage: "", message: agent.role, ts: null,
  }]));
  state.liveEvents = [];
  state.liveStages = [];
  state.behaviorEvents = [];
  renderAgentActivity();
  renderAgentBehaviors([]);
  setLiveState("idle", "等待运行");
}
function jsonPreview(value) {
  if (value == null) return "";
  if (Array.isArray(value) && !value.length) return "";
  if (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length) return "";
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}
function renderAgentBehaviors(events) {
  if (Array.isArray(events)) state.behaviorEvents = events.slice();
  const allRows = state.behaviorEvents;
  const selectedAgent = $("behavior-agent-filter")?.value || "all";
  const selectedType = $("behavior-type-filter")?.value || "all";
  const selectedStatus = $("behavior-status-filter")?.value || "all";
  const agentNames = [...new Set([...AGENTS.map((agent) => agent.name), ...allRows.map((event) => event.agent).filter(Boolean)])];
  const agentSelect = $("behavior-agent-filter");
  if (agentSelect) {
    agentSelect.innerHTML = '<option value="all">全部角色</option>' + agentNames.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    agentSelect.value = agentNames.includes(selectedAgent) ? selectedAgent : "all";
  }
  const rows = allRows.filter((event) => {
    const type = event.type === "manager_decision" ? "decision" : "action";
    return (selectedAgent === "all" || event.agent === selectedAgent)
      && (selectedType === "all" || type === selectedType)
      && (selectedStatus === "all" || event.status === selectedStatus);
  });
  const decisions = allRows.filter((event) => event.type === "manager_decision").length;
  const completed = allRows.filter((event) => event.type !== "manager_decision" && event.status && event.status !== "running").length;
  const agents = new Set(allRows.map((event) => event.agent).filter(Boolean));
  $("behavior-summary").textContent = allRows.length
    ? `${agents.size} 个角色 · ${completed} 次动作完成 · ${decisions} 次 Manager 路由决策 · 当前显示 ${rows.length}/${allRows.length}`
    : "暂无 Agent 行为记录";
  const stream = $("agent-behavior-list");
  stream.innerHTML = rows.length ? rows.map((event, index) => {
    const decision = event.decision || {};
    const isDecision = event.type === "manager_decision";
    const agent = AGENTS.find((item) => item.name === event.agent) || AGENTS[0];
    const toolCalls = Array.isArray(event.tool_calls) ? event.tool_calls : [];
    const input = jsonPreview(event.input_summary || event.payload || {});
    const output = jsonPreview(event.output_summary || {});
    const detailLabel = { info: "运行细节", success: "步骤完成", warning: "校验 / 重试" }[event.detail_status];
    const feedback = decision.feedback ? `<div><span>反馈</span><p>${escapeHtml(decision.feedback)}</p></div>` : "";
    const detail = isDecision
      ? `<div class="behavior-route"><span>${escapeHtml(decision.after_stage || event.stage || "--")}</span><b>→</b><strong>${escapeHtml(decision.next_agent || decision.next_node || "--")}</strong></div>
         <div class="behavior-fields"><div><span>原因码</span><code>${escapeHtml(decision.reason_code || "--")}</code></div><div><span>剩余修订预算</span><strong>${escapeHtml(decision.remaining_revision_budget ?? "--")}</strong></div>${feedback}</div>`
      : `<div class="behavior-fields"><div><span>动作</span><code>${escapeHtml(event.action || event.stage || "--")}</code></div><div><span>尝试</span><strong>${escapeHtml(event.attempt || 1)}</strong></div><div><span>耗时</span><strong>${event.duration_ms == null ? "--" : `${formatNumber(event.duration_ms, 1)} ms`}</strong></div></div>`;
    const payloads = [
      input ? `<details><summary>输入摘要</summary><pre>${escapeHtml(input)}</pre></details>` : "",
      output ? `<details><summary>输出摘要</summary><pre>${escapeHtml(output)}</pre></details>` : "",
      toolCalls.length ? `<details><summary>工具调用 ${toolCalls.length}</summary><pre>${escapeHtml(jsonPreview(toolCalls))}</pre></details>` : "",
      isDecision ? `<details><summary>完整路由决策</summary><pre>${escapeHtml(jsonPreview(decision))}</pre></details>` : "",
    ].join("");
    return `<article class="behavior-card ${isDecision ? "manager-route" : escapeHtml(event.status || "pending")} ${event.detail ? `detail-${escapeHtml(event.detail_status || "info")}` : ""}" data-behavior-index="${index}">
      <div class="behavior-avatar">${escapeHtml(agent.icon)}</div><div class="behavior-body">
      <header><div><span>${isDecision ? "Manager 路由" : escapeHtml(detailLabel || STATUS_LABELS[event.status] || "Agent 动作")}</span><strong>${escapeHtml(event.agent || "Agent Manager")}</strong></div><time>${escapeHtml(formatTime(event.completed_at || event.ts))}</time></header>
      <p>${escapeHtml(event.message || "无行为说明")}</p>${detail}${payloads}</div>
    </article>`;
  }).join("") : '<div class="empty-panel">启动真实 Agent 流程后显示工作消息。</div>';
  if ($("agent-autoscroll")?.checked) stream.scrollTop = stream.scrollHeight;
}
function exportAgentAudit() {
  if (!state.behaviorEvents.length) return toast("暂无可导出的 Agent 行为", true);
  const payload = {
    schema_version: "yieldmind-agent-audit-v1",
    exported_at: new Date().toISOString(),
    run_id: state.currentRunId || null,
    workflow_kind: state.result?.manager_decisions ? "domain" : "offline",
    events: state.behaviorEvents,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.currentRunId || "yieldmind"}_agent_audit.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
function renderAgentActivity() {
  $("agent-grid").innerHTML = AGENTS.map((agent) => {
    const activity = state.agentStates[agent.name] || { status: "pending", message: agent.role };
    return `<button class="agent-card ${escapeHtml(activity.status)}" type="button" data-agent-filter="${escapeHtml(agent.name)}">
      <div class="agent-card-head"><strong><span class="agent-mini-avatar">${escapeHtml(agent.icon)}</span>${escapeHtml(agent.name)}</strong><span class="agent-status">${escapeHtml(STATUS_LABELS[activity.status] || activity.status)}</span></div>
      <p>${escapeHtml(activity.message || agent.role)}</p>
      <small>${escapeHtml(activity.stage ? `${STAGE_LABELS[activity.stage] || activity.stage} · ${formatTime(activity.ts)}` : agent.role)}</small>
    </button>`;
  }).join("");
  $("activity-count").textContent = `${state.liveEvents.length} events`;
  document.querySelectorAll("[data-agent-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      const select = $("behavior-agent-filter");
      select.value = select.value === button.dataset.agentFilter ? "all" : button.dataset.agentFilter;
      renderAgentBehaviors();
    });
  });
}
function renderLivePipeline() {
  renderPipeline({
    stages: state.liveStages,
    workflow_backend: "workflow_stream",
    checkpoint_backend: "running",
    local_repair_count: state.liveStages.filter((item) => item.stage === "local_repair" && item.status !== "running").length,
    replan_count: state.liveStages.filter((item) => item.stage === "replan" && item.status !== "running").length,
  });
}
function handleAgentProgress(event) {
  const current = state.agentStates[event.agent] || { status: "pending" };
  state.agentStates[event.agent] = { ...current, status: event.status, stage: event.stage, message: event.message, ts: event.ts };
  state.liveEvents.push(event);
  if (state.liveEvents.length > 80) state.liveEvents.shift();
  if (event.detail) {
    // Runtime details enrich the scrolling Agent transcript without creating
    // a duplicate StateGraph node for every code-generation log line.
  } else if (event.status === "running") {
    state.liveStages.push({ stage: event.stage, status: "running", message: event.message, ts: event.ts });
  } else {
    const index = state.liveStages.map((item) => `${item.stage}:${item.status}`).lastIndexOf(`${event.stage}:running`);
    const completed = { stage: event.stage, status: event.status, message: event.message, ts: event.ts };
    if (index >= 0) state.liveStages[index] = completed; else state.liveStages.push(completed);
  }
  if (event.run_id) {
    state.currentRunId = event.run_id;
    $("run-title").textContent = `领域多 Agent · ${event.run_id}`;
  }
  $("run-subtitle").textContent = `${event.agent} · ${STAGE_LABELS[event.stage] || event.stage}`;
  $("run-feedback").textContent = `${event.agent}：${event.message}`;
  renderAgentActivity();
  renderAgentBehaviors(state.liveEvents);
  renderLivePipeline();
}
function normalizeOperationEvent(event, result) {
  if (event?.stage !== "operation" || event?.status === "running" || event?.detail) return event;
  const operation = event?.output_summary?.operation || result?.operation_result || {};
  const rcode = Number(operation.rcode);
  if (operation.cancelled) {
    return { ...event, status: "cancelled", message: operation.action_result || event.message };
  }
  if (operation.timed_out || (Number.isFinite(rcode) && rcode !== 0)) {
    return { ...event, status: "failed", message: operation.action_result || "OperationAgent 执行失败。" };
  }
  return event;
}
function reconcileAgentStates(result) {
  (result?.stages || []).forEach((stage) => {
    stage = normalizeOperationEvent(stage, result);
    const agent = stage.agent || "Agent Manager";
    if (!state.agentStates[agent]) return;
    state.agentStates[agent] = {
      ...state.agentStates[agent], status: stage.status, stage: stage.stage,
      message: stage.message, ts: stage.completed_at || stage.ts,
    };
  });
  AGENTS.forEach((agent) => {
    const current = state.agentStates[agent.name];
    if (current?.status === "running") {
      state.agentStates[agent.name] = {
        ...current,
        status: result?.status === "cancelled" ? "cancelled" : result?.status === "failed" ? "failed" : "passed",
        message: `工作流已进入终态：${result?.status || "unknown"}`,
      };
    } else if (current?.status === "pending") {
      state.agentStates[agent.name] = { ...current, status: "skipped", message: "本次真实流程未调用该 Agent。" };
    }
  });
  renderAgentActivity();
}
function finalizeAgentStates(status, message) {
  AGENTS.forEach((agent) => {
    const current = state.agentStates[agent.name] || { status: "pending", message: agent.role };
    if (current.status === "running") {
      state.agentStates[agent.name] = { ...current, status, message };
    } else if (current.status === "pending") {
      state.agentStates[agent.name] = { ...current, status: "skipped", message: "流程结束前未调用该 Agent。" };
    }
  });
  renderAgentActivity();
}
function handleManagerDecision(event) {
  const decision = event.decision || {};
  const message = event.message || `路由到 ${decision.next_agent || decision.next_node || "下一节点"}`;
  const current = state.agentStates["Agent Manager"] || { status: "pending" };
  state.agentStates["Agent Manager"] = {
    ...current, status: event.status || "passed", stage: event.stage || decision.after_stage || "",
    message, ts: event.ts,
  };
  state.liveEvents.push({ ...event, agent: "Agent Manager", message });
  if (state.liveEvents.length > 80) state.liveEvents.shift();
  $("run-feedback").textContent = `Agent Manager：${message}`;
  renderAgentActivity();
  renderAgentBehaviors(state.liveEvents);
}
function renderAgentHistory(result) {
  resetAgentActivity();
  if (!isDomainResult(result)) {
    state.agentStates = Object.fromEntries(AGENTS.map((agent) => [agent.name, {
      status: "skipped", stage: "", message: "该记录是离线基线评测，不是 Agent 执行。", ts: null,
    }]));
    renderAgentActivity();
    renderAgentBehaviors([]);
    setLiveState("idle", "离线评测记录");
    return;
  }
  const agentByStage = {
    start: "Agent Manager", prepare: "Agent Manager", data: "DataAgent", requirements: "Agent Manager",
    search: "SearchAgent", candidate: "CandidateAgent", model: "ModelAgent", pre_execution: "Agent Manager",
    operation: "OperationAgent", review: "Agent Manager", revision_feedback: "Agent Manager",
    cancel: "Agent Manager", finish: "Agent Manager",
  };
  const activities = result?.agent_activities?.length ? result.agent_activities : (result?.stages || []);
  activities.forEach((stage) => {
    if (stage.type === "manager_decision") {
      handleManagerDecision(stage);
      return;
    }
    const event = normalizeOperationEvent(
      { type: "agent_progress", agent: agentByStage[stage.stage] || "Agent Manager", ...stage },
      result,
    );
    const current = state.agentStates[event.agent];
    state.agentStates[event.agent] = { ...current, status: event.status, stage: event.stage, message: event.message, ts: event.ts };
    state.liveEvents.push(event);
  });
  reconcileAgentStates(result);
  renderAgentActivity();
  renderAgentBehaviors(state.liveEvents);
  setLiveState(result?.status || "idle", STATUS_LABELS[result?.status] || result?.status || "等待运行");
}

function setDataset(data, label) {
  $("data-path").value = data.data_path || "";
  $("dataset-kind").textContent = data.dataset_role_label || label;
  $("dataset-name").textContent = data.original_filename || data.filename || "未选择 CSV";
  const boundary = data.evaluation_boundary ? ` · ${data.evaluation_boundary}` : "";
  $("dataset-meta").textContent = `${formatNumber(data.row_count, 0)} 行 · ${formatNumber(data.column_count, 0)} 列 · 目标列 ${data.target_column || "yield_stress"}${boundary}`;
}
async function loadDefaultDataset() {
  try {
    setDataset(await requestJson("/api/data/default"), "默认增强演示数据");
  } catch (error) {
    $("dataset-kind").textContent = "默认数据不可用";
    $("dataset-name").textContent = "可上传 CSV 或留空生成演示数据";
    $("dataset-meta").textContent = error.message;
  }
}
async function uploadCsv() {
  const file = $("csv-upload").files[0];
  if (!file) return toast("请先选择一个 CSV 文件", true);
  const button = $("upload-csv");
  button.disabled = true;
  button.textContent = "正在校验...";
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await requestJson("/api/data/upload", { method: "POST", body: form });
    setDataset(data, "用户上传数据");
    toast(`已载入 ${data.original_filename}，共 ${data.row_count} 条有效样本`);
  } catch (error) {
    toast(`上传失败：${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "上传并使用 CSV";
  }
}

async function refreshHealth() {
  try {
    const [health, dependencies] = await Promise.all([
      requestJson("/health"),
      fetch("/health/dependencies").then((response) => response.json()),
    ]);
    setStatus("status-api", health.ok, "API");
    setStatus("status-db", dependencies.database?.ok, dependencies.database?.backend || "Database");
    setStatus("status-redis", dependencies.redis?.ok, "Redis");
    setStatus("status-knowledge", dependencies.knowledge?.ok, dependencies.knowledge?.profile || "Knowledge");
    ["enqueue-task", "refresh-task", "cancel-task", "list-stale", "recover-task"].forEach((id) => { $(id).disabled = !dependencies.redis?.ok; });
    $("queue-section").title = dependencies.redis?.ok ? "" : "Redis 未就绪，任务队列操作暂不可用";
  } catch (error) {
    ["status-api", "status-db", "status-redis", "status-knowledge"].forEach((id) => setStatus(id, false));
    toast(`健康检查失败：${error.message}`, true);
  }
}

function toolPayload(result, name) {
  const item = (result?.tool_results || []).find((entry) => entry.tool === name);
  return item?.result?.result || {};
}
function selectedCandidate(benchmark) {
  return (benchmark?.strategy_benchmarks || []).find((row) => row.strategy_id === benchmark.selected_strategy_id) || null;
}
function finalSelection(benchmark) {
  const selection = benchmark?.final_selection;
  if (selection && Object.keys(selection).length) return selection;
  const candidate = selectedCandidate(benchmark);
  return candidate ? {
    winner_type: "candidate_strategy", winner_id: candidate.strategy_id,
    winner_name: candidate.strategy_name || candidate.strategy_id,
    winner_oof_metrics: candidate.oof_metrics || {},
    winner_prediction_preview: benchmark?.selected_strategy_prediction_preview || [],
    status: "legacy_candidate_selection", reason: "历史运行未保存全局选择结果。",
  } : {};
}
function finalSelectionReason(selection) {
  return FINAL_SELECTION_REASONS[selection?.status] || selection?.reason || "尚无全局选择结果";
}
function isDomainResult(result) {
  return Boolean(result?.manager_args || result?.manager_decisions || result?.workflow_kind === "domain");
}
function setRunHeader(result, source = "运行结果") {
  const runId = result?.run_id || "unknown";
  state.currentRunId = runId;
  const domain = isDomainResult(result);
  $("run-title").textContent = `${domain ? "领域多 Agent" : "屈服应力实验"} · ${runId}`;
  const dataSource = domain
    ? `领域链路 · ${result?.model_call_mode || "unknown model mode"}`
    : result?.data_source === "provided" ? "指定 CSV" : "确定性演示数据";
  $("run-subtitle").textContent = `${source} · ${dataSource} · ${result?.thread_id || "no thread"}`;
  const status = result?.status || "idle";
  $("run-state").textContent = status;
  $("run-state").dataset.state = status;
}
function setMetricSlot(key, label, value, note) {
  $(`metric-label-${key}`).textContent = label;
  $(`metric-${key}`).textContent = value;
  $(`metric-note-${key}`).textContent = note;
}
function renderMetrics(metrics = {}, result = {}) {
  if (isDomainResult(result)) {
    const completed = (result.stages || []).filter((stage) => stage.status !== "running").length;
    setMetricSlot("r2", "修订轮次", formatNumber(result.manager_revision_round || 0, 0), `预算 ${formatNumber(result.max_manager_revisions || 0, 0)}`);
    setMetricSlot("rmse", "Manager 决策", formatNumber((result.manager_decisions || []).length, 0), "全部通过合法边校验");
    setMetricSlot("mae", "完成节点", formatNumber(completed, 0), `共 ${(result.stages || []).length} 条阶段记录`);
    setMetricSlot("mape", "模型调用模式", result.model_call_mode || "--", result.real_llm_calls == null ? "调用数未统一计量" : `真实调用 ${result.real_llm_calls} 次`);
    return;
  }
  setMetricSlot("r2", "OOF R²", formatNumber(metrics.r2), "最终推荐");
  setMetricSlot("rmse", "OOF RMSE", formatNumber(metrics.rmse), "越低越好");
  setMetricSlot("mae", "OOF MAE", formatNumber(metrics.mae), "绝对误差");
  setMetricSlot("mape", "OOF MAPE", metrics.mape == null ? "--" : `${formatNumber(metrics.mape, 2)}%`, "百分比误差");
}
function renderDomainOverview(result) {
  const operation = result?.operation_result || {};
  const review = result?.post_execution_review || {};
  const decisions = result?.manager_decisions || [];
  const lastDecision = decisions.length ? decisions[decisions.length - 1] : {};
  const process = result?.process_control || {};
  $("domain-result-mode").textContent = `${result?.execution_mode || "--"} · ${result?.model_call_mode || "--"}`;
  $("domain-decision-audit").innerHTML = `
    <div class="decision-block"><span>执行前审批</span><strong>${result?.pre_execution_passed ? "已通过" : "未通过 / 未执行"}</strong><p>${result?.pre_execution_passed ? "OperationAgent 获得执行许可。" : "流程不会绕过 Manager 审批。"}</p></div>
    <div class="decision-block"><span>Operation 结果</span><strong>rcode ${escapeHtml(operation.rcode ?? "--")}</strong><p>${escapeHtml(operation.action_result || operation.stage || "暂无 Operation 结果")}</p></div>
    <div class="decision-block"><span>Manager Review</span><strong>${review.passed ? "accepted" : escapeHtml(review.decision || "not accepted")}</strong><p>${escapeHtml(review.reason || (review.issues || []).join("；") || "暂无 review 说明")}</p></div>
    <div class="decision-block"><span>最后路由</span><strong>${escapeHtml(lastDecision.after_stage || "--")} → ${escapeHtml(lastDecision.next_agent || lastDecision.next_node || "--")}</strong><p>${escapeHtml(lastDecision.reason_code || "暂无结构化路由")}</p></div>
    <div class="decision-block"><span>修订反馈</span><strong>${result?.manager_feedback ? "已生成" : "无"}</strong><p>${escapeHtml(result?.manager_feedback || "当前运行没有进入修订回环。")}</p></div>
    <div class="decision-block"><span>进程控制</span><strong>${escapeHtml(process.status || (result?.cancelled ? "cancelled" : "--"))}</strong><p>${escapeHtml(process.termination_signal || process.error || "未触发额外终止信号。")}</p></div>`;
}
function configureResultMode(result) {
  const domain = isDomainResult(result);
  $("offline-analysis").hidden = domain;
  $("domain-overview").hidden = !domain;
  ["baselines", "decision"].forEach((name) => {
    const tab = document.querySelector(`.tab[data-tab="${name}"]`);
    if (tab) tab.hidden = domain;
  });
  if (domain) {
    renderDomainOverview(result);
    activateTab("artifacts");
  } else if (document.querySelector(".tab.active")?.hidden) {
    activateTab("baselines");
  }
}
function renderPipeline(result) {
  const stages = result?.stages || [];
  $("workflow-backend").textContent = `${result?.workflow_backend || "--"} · ${result?.checkpoint_backend || "--"}`;
  const passed = stages.filter((stage) => stage.status === "passed").length;
  $("pipeline-summary").textContent = stages.length
    ? isDomainResult(result)
      ? `${passed}/${stages.length} 节点通过 · Manager 修订 ${result?.manager_revision_round || 0} 次 · 路由决策 ${(result?.manager_decisions || []).length} 条`
      : `${passed}/${stages.length} 节点通过 · 修复 ${result?.local_repair_count || 0} 次 · 重规划 ${result?.replan_count || 0} 次`
    : "暂无节点记录";
  $("pipeline").innerHTML = stages.length ? stages.map((stage, index) => `
    <div class="pipeline-node ${escapeHtml(stage.status)}" title="${escapeHtml(stage.message)}">
      <span class="node-index">${String(index + 1).padStart(2, "0")} · ${escapeHtml(stage.status)}</span>
      <strong>${escapeHtml(STAGE_LABELS[stage.stage] || stage.stage)}</strong><small>${escapeHtml(stage.message)}</small>
    </div>`).join("") : '<div class="empty-panel">暂无节点记录</div>';
}
function renderCandidates(benchmark) {
  const rows = benchmark?.strategy_benchmarks || [];
  const selectedId = benchmark?.selected_strategy_id;
  const final = finalSelection(benchmark);
  const baselineWins = final.winner_type === "fixed_baseline";
  $("candidate-count").textContent = `${rows.length} candidates`;
  $("selection-summary").textContent = final.winner_id
    ? baselineWins
      ? `最终推荐固定基线 ${final.winner_id}；候选优胜者 ${selectedId || "--"} 未达到更低 OOF RMSE。`
      : `最终推荐候选 ${final.winner_id}；它在同折比较中优于最佳固定基线。`
    : "候选与固定基线暂无同协议可比结果，未产生最终推荐。";
  $("candidate-table").innerHTML = rows.length ? rows.map((row) => {
    const selected = row.strategy_id === selectedId;
    const metrics = row.oof_metrics || {};
    const mechanisms = (row.mechanism_ids || []).join(", ") || "无机理特征";
    const failed = row.benchmark_status !== "evaluated";
    return `<tr class="${selected && !baselineWins ? "selected" : ""}">
      <td class="candidate-name"><strong>${escapeHtml(row.strategy_name || row.strategy_id)}</strong><small>${escapeHtml(row.strategy_id)}</small></td>
      <td class="model-stack">${escapeHtml(row.proxy_model || "--")}<small>${escapeHtml(mechanisms)}</small></td>
      <td>${formatNumber(metrics.rmse)}</td><td>${formatNumber(metrics.r2)}</td><td>${formatNumber(row.audited_selection_score_1_to_10, 3)}</td>
      <td><span class="result-badge ${selected && !baselineWins ? "selected" : failed ? "failed" : ""}">${selected ? baselineWins ? "候选优胜" : "最终推荐" : failed ? "未评估" : "候选"}</span></td>
    </tr>`;
  }).join("") : '<tr><td colspan="6" class="empty-table">暂无候选评测</td></tr>';
}
function renderBaselines(baseline, benchmark) {
  const report = baseline?.report || {};
  const rows = report.baseline_results || [];
  const best = report.best_baseline;
  const final = finalSelection(benchmark);
  $("baseline-table").innerHTML = rows.length ? rows.map((row) => `
    <tr class="${row.name === final.winner_id ? "selected" : ""}">
      <td class="candidate-name"><strong>${escapeHtml(row.name)}</strong><small>${row.name === final.winner_id ? "最终推荐" : row.name === best ? "最佳固定基线" : escapeHtml(row.notes || "")}</small></td>
      <td>${formatNumber(row.oof_rmse)}</td><td>${formatNumber(row.oof_mae)}</td><td>${formatNumber(row.oof_r2)}</td><td>${formatNumber(row.oof_mape, 2)}%</td>
    </tr>`).join("") : '<tr><td colspan="5" class="empty-table">暂无基线结果</td></tr>';
}
function renderDecision(benchmark, profile) {
  const selected = selectedCandidate(benchmark) || {};
  const final = finalSelection(benchmark);
  const baselineRmse = benchmark?.best_fixed_baseline_oof_rmse;
  const candidateRmse = benchmark?.selected_strategy_oof_rmse;
  const delta = Number.isFinite(Number(final.candidate_minus_baseline_oof_rmse)) ? Number(final.candidate_minus_baseline_oof_rmse) : null;
  const viability = benchmark?.anchor_viability_summary || {};
  $("decision-audit").innerHTML = `
    <div class="decision-block"><span>数据</span><strong>${formatNumber(profile?.row_count, 0)} 行 · ${formatNumber(profile?.column_count, 0)} 列</strong><p>目标范围 ${formatNumber(profile?.target_summary?.min)} 到 ${formatNumber(profile?.target_summary?.max)}</p></div>
    <div class="decision-block"><span>最终推荐</span><strong>${escapeHtml(final.winner_name || final.winner_id || "--")}</strong><p>${escapeHtml(finalSelectionReason(final))}</p></div>
    <div class="decision-block"><span>候选相对固定基线</span><strong>${delta == null ? "--" : delta === 0 ? "OOF RMSE 持平" : `${delta < 0 ? "降低" : "增加"} ${formatNumber(Math.abs(delta))} RMSE`}</strong><p>候选 ${formatNumber(candidateRmse)} / 基线 ${formatNumber(baselineRmse)}</p></div>
    <div class="decision-block"><span>可行候选</span><strong>${formatNumber(viability.viable_strategy_count, 0)} / ${formatNumber(viability.evaluated_strategy_count, 0)}</strong><p>${escapeHtml(viability.failure_reason || "通过当前可行性门槛")}</p></div>
    <div class="decision-block"><span>候选优胜者</span><strong>${escapeHtml(selected.strategy_name || selected.strategy_id || "--")}</strong><p>${escapeHtml(selected.anchor_viability_status || benchmark?.selected_strategy_status || "未声明")}</p></div>
    <div class="decision-block"><span>模型调用</span><strong>0 次真实 LLM</strong><p>结果来自确定性 sklearn 与候选代理评测。</p></div>`;
}
function renderArtifacts(result) {
  const entries = Object.entries(result?.artifacts || {});
  $("artifact-list").innerHTML = entries.length ? entries.map(([name, path]) => `<div class="artifact-row"><strong>${escapeHtml(name)}</strong><code>${escapeHtml(path)}</code><span>${escapeHtml(String(path).split(".").pop().toUpperCase())}</span></div>`).join("") : '<div class="empty-panel">暂无运行产物</div>';
}
function renderTrace(result) {
  const rows = result?.stages || [];
  $("trace-list").innerHTML = rows.length ? rows.map((row) => `<div class="trace-row"><strong>${escapeHtml(row.stage)}</strong><p>${escapeHtml(row.message)}<br>input ${escapeHtml(row.input_hash || "--")}</p><span>${escapeHtml(row.status)}</span></div>`).join("") : '<div class="empty-panel">暂无 Trace</div>';
}

function renderPredictionChart(points) {
  state.chartPoints = Array.isArray(points) ? points : [];
  const canvas = $("prediction-chart");
  const empty = $("chart-empty");
  const context = canvas.getContext("2d");
  if (!state.chartPoints.length) {
    empty.style.display = "grid";
    context.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  empty.style.display = "none";
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const margin = { left: 49, right: 20, top: 20, bottom: 42 };
  const values = state.chartPoints.flatMap((point) => [Number(point.y_true), Number(point.y_pred)]).filter(Number.isFinite);
  let min = Math.min(...values);
  let max = Math.max(...values);
  const padding = Math.max((max - min) * 0.08, 0.05);
  min = Math.max(0, min - padding);
  max += padding;
  const x = (value) => margin.left + ((value - min) / (max - min || 1)) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - ((value - min) / (max - min || 1)) * (height - margin.top - margin.bottom);
  context.clearRect(0, 0, width, height);
  context.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
  context.fillStyle = "#68716b";
  context.strokeStyle = "#e2e6e2";
  context.lineWidth = 1;
  for (let step = 0; step <= 4; step += 1) {
    const value = min + ((max - min) * step) / 4;
    const px = x(value); const py = y(value);
    context.beginPath(); context.moveTo(margin.left, py); context.lineTo(width - margin.right, py); context.stroke();
    context.beginPath(); context.moveTo(px, margin.top); context.lineTo(px, height - margin.bottom); context.stroke();
    context.fillText(formatNumber(value, 2), 5, py + 3);
    context.fillText(formatNumber(value, 2), px - 10, height - 20);
  }
  context.strokeStyle = "#176b4d";
  context.lineWidth = 1.5;
  context.beginPath(); context.moveTo(x(min), y(min)); context.lineTo(x(max), y(max)); context.stroke();
  context.fillStyle = "rgba(40, 100, 165, .72)";
  state.chartPoints.forEach((point) => { context.beginPath(); context.arc(x(Number(point.y_true)), y(Number(point.y_pred)), 3.2, 0, Math.PI * 2); context.fill(); });
  context.fillStyle = "#4f5b54";
  context.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  context.fillText("实际值", width / 2 - 15, height - 5);
  context.save(); context.translate(12, height / 2 + 15); context.rotate(-Math.PI / 2); context.fillText("预测值", 0, 0); context.restore();
}

function renderResult(result, source = "运行结果", preserveLiveActivity = false) {
  state.result = result;
  setRunHeader(result, source);
  configureResultMode(result);
  if (isDomainResult(result)) {
    if (!preserveLiveActivity) renderAgentHistory(result);
    else {
      setLiveState(result?.status || "idle", STATUS_LABELS[result?.status] || result?.status);
      renderAgentBehaviors(result?.agent_activities?.length ? result.agent_activities : state.liveEvents);
      reconcileAgentStates(result);
    }
  } else {
    renderAgentHistory(result);
  }
  const baseline = toolPayload(result, "run_fixed_baseline_eval");
  const candidatePayload = toolPayload(result, "run_candidate_benchmark");
  const benchmark = candidatePayload.benchmark_report || {};
  const profile = toolPayload(result, "profile_yield_data");
  const selected = selectedCandidate(benchmark);
  const final = finalSelection(benchmark);
  renderMetrics(final.winner_id ? (final.winner_oof_metrics || {}) : {}, result);
  renderPipeline(result);
  renderCandidates(benchmark);
  renderBaselines(baseline, benchmark);
  renderDecision(benchmark, profile);
  renderArtifacts(result);
  renderTrace(result);
  $("raw-output").textContent = JSON.stringify(result, null, 2);
  $("prediction-caption").textContent = final.winner_id
    ? `${final.winner_name || final.winner_id} · ${final.comparison_protocol || benchmark.benchmark_protocol?.primary_evaluation || "OOF"}`
    : "暂无最终推荐";
  window.requestAnimationFrame(() => renderPredictionChart(final.winner_prediction_preview || benchmark.selected_strategy_prediction_preview || []));
  highlightRecentRun(result.run_id);
}

async function streamWorkflow(endpoint, body) {
  const response = await fetch(endpoint, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* response was not JSON */ }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("浏览器未提供流式响应读取能力");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "agent_progress") handleAgentProgress(event);
      if (event.type === "manager_decision") handleManagerDecision(event);
      if (event.type === "result") finalResult = event.result;
      if (event.type === "error") throw new Error(event.error || "工作流执行失败");
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "agent_progress") handleAgentProgress(event);
    if (event.type === "manager_decision") handleManagerDecision(event);
    if (event.type === "result") finalResult = event.result;
    if (event.type === "error") throw new Error(event.error || "工作流执行失败");
  }
  if (!finalResult) throw new Error("工作流已结束，但未返回最终结果");
  return finalResult;
}

async function runWorkflow() {
  if (!$("allow-live-llm").checked) {
    return toast("请先确认调用真实 LLM 并执行生成代码。", true);
  }
  const button = $("run-workflow");
  button.disabled = true;
  button.textContent = "Agent 运行中...";
  $("run-feedback").textContent = "正在连接真实领域 Agent 事件流；该过程可能持续数分钟。";
  $("run-state").textContent = "running";
  $("run-state").dataset.state = "running";
  resetAgentActivity();
  setLiveState("running", "实时执行中");
  try {
    const prompt = $("prompt").value.trim();
    const session = await ensureSessionTurn(prompt);
    const body = {
      prompt, data_path: $("data-path").value.trim(),
      llm: $("domain-llm").value,
      n_revise: Number($("revision-rounds").value || 0),
      operation_attempts: Number($("operation-attempts").value || 3),
      execution_mode: $("execution-mode").value,
      use_knowledge_search: $("domain-knowledge-search").checked,
      knowledge_top_k: 5,
      knowledge_context_budget_chars: 12000,
      external_search: $("external-search").checked,
      require_search_results: false,
      synthetic_data: null,
      allow_live_llm: true,
      workspace_id: "yieldmind_web",
      session_id: session.sessionId,
      turn_id: session.turnId,
      session_context_max_tokens: 1200,
    };
    const result = await streamWorkflow("/api/workflows/domain/stream", body);
    renderResult(result, "新运行", true);
    $("run-feedback").textContent = `运行完成：${result.status}`;
    toast(`运行 ${result.run_id} 已完成`);
    await loadRecentRuns();
  } catch (error) {
    $("run-feedback").textContent = `运行失败：${error.message}`;
    $("run-state").textContent = "failed";
    $("run-state").dataset.state = "failed";
    setLiveState("failed", "执行失败");
    finalizeAgentStates("failed", `工作流失败：${error.message}`);
    toast(`运行失败：${error.message}`, true);
  } finally { button.disabled = false; button.textContent = "启动真实 Agent 流程"; }
}

async function runOfflineEvaluation() {
  const button = $("run-offline-eval");
  button.disabled = true;
  button.textContent = "评测中...";
  $("offline-eval-feedback").textContent = "正在运行确定性基线；不会产生 Agent 消息。";
  try {
    const useKnowledge = $("use-knowledge").checked;
    const result = await requestJson("/api/workflows/offline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value.trim(), data_path: $("data-path").value.trim(),
        n_samples: Number($("samples").value || 60), n_splits: Number($("folds").value || 3),
        use_knowledge: useKnowledge,
        knowledge_query: useKnowledge ? $("knowledge-query").value.trim() : "",
        knowledge_retrieval_mode: $("retrieval-mode").value,
      }),
    });
    renderResult(result, "离线基线评测");
    $("offline-eval-feedback").textContent = `评测完成：${result.status}（非 Agent 流程）`;
    toast(`离线评测 ${result.run_id} 已完成`);
    await loadRecentRuns();
  } catch (error) {
    $("offline-eval-feedback").textContent = `评测失败：${error.message}`;
    toast(`离线评测失败：${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "运行离线评测";
  }
}

async function loadRecentRuns() {
  try {
    const data = await requestJson("/api/runs?limit=10");
    const runs = data.runs || [];
    $("recent-runs").innerHTML = runs.length ? runs.map((run) => `
      <button class="recent-run ${run.run_id === state.currentRunId ? "active" : ""}" type="button" data-run-id="${escapeHtml(run.run_id)}" data-source="${escapeHtml(run.source || "")}">
        <span class="dot ${escapeHtml(run.status)}"></span><span><strong>${escapeHtml(run.run_id)}</strong><small>${escapeHtml(run.source)} · ${formatTime(run.started_at)}</small></span><small>${escapeHtml(run.status)}</small>
      </button>`).join("") : '<div class="empty-compact">暂无运行记录</div>';
    document.querySelectorAll(".recent-run").forEach((button) => { button.addEventListener("click", () => loadRun(button.dataset.runId)); });
  } catch (error) { $("recent-runs").innerHTML = `<div class="empty-compact">加载失败：${escapeHtml(error.message)}</div>`; }
}
function highlightRecentRun(runId) {
  document.querySelectorAll(".recent-run").forEach((button) => button.classList.toggle("active", button.dataset.runId === runId));
}
async function loadRun(runId) {
  try {
    const data = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
    if (!data.run?.result || !Object.keys(data.run.result).length) throw new Error("该运行没有可展示的结果");
    renderResult(data.run.result, "历史运行");
  } catch (error) { toast(`加载运行失败：${error.message}`, true); }
}
async function searchKnowledge() {
  try {
    const data = await requestJson("/api/knowledge/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: $("quick-query").value.trim(), top_k: 5, retrieval_mode: "hybrid", routing_mode: "auto", expand_parent: true, deduplicate_parents: true, context_budget_chars: 12000 }) });
    $("raw-output").textContent = JSON.stringify(data, null, 2);
    activateTab("raw");
    toast(`知识检索返回 ${(data.hits || []).length} 条结果`);
  } catch (error) { toast(`知识检索失败：${error.message}`, true); }
}
async function loadTools() {
  try {
    const data = await requestJson("/api/tools");
    $("tool-select").innerHTML = (data.tools || []).map((tool) => `<option value="${escapeHtml(tool.name)}">${escapeHtml(tool.name)}</option>`).join("");
  } catch (error) { toast(`工具列表加载失败：${error.message}`, true); }
}
async function callTool() {
  try {
    let args;
    try { args = JSON.parse($("tool-args").value || "{}"); } catch (error) { throw new Error(`参数 JSON 错误：${error.message}`); }
    const name = $("tool-select").value;
    const data = await requestJson(`/api/tools/${encodeURIComponent(name)}/call`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_id: state.currentRunId || null, args }) });
    $("raw-output").textContent = JSON.stringify(data, null, 2);
    activateTab("raw");
  } catch (error) { toast(error.message, true); }
}

function renderTask(data) {
  const task = data.task || {};
  $("task-box").innerHTML = `<strong>${escapeHtml(task.task_id || state.currentTaskId)}</strong><br>DB: ${escapeHtml(task.status || "unknown")} · Celery: ${escapeHtml(data.celery_state || "-")}<br>${escapeHtml(task.cancel_reason || task.recovery_reason || "")}`;
  if (["completed", "failed", "dispatch_failed", "cancelled"].includes(task.status) && state.taskPollTimer) { window.clearInterval(state.taskPollTimer); state.taskPollTimer = null; }
}
async function enqueueTask() {
  try {
    const key = `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const data = await requestJson("/api/tasks/workflows/offline", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workflow: { prompt: "Queued workflow from YieldMind console.", n_samples: Number($("task-samples").value || 40), n_splits: 2 }, idempotency_key: key }) });
    state.currentTaskId = data.task.task_id;
    renderTask({ task: data.task, celery_state: "PENDING" });
    if (state.taskPollTimer) window.clearInterval(state.taskPollTimer);
    state.taskPollTimer = window.setInterval(refreshTask, 1000);
  } catch (error) { toast(`任务提交失败：${error.message}`, true); }
}
async function refreshTask() {
  if (!state.currentTaskId) return toast("尚无 task_id", true);
  try { renderTask(await requestJson(`/api/tasks/${state.currentTaskId}`)); } catch (error) { toast(`任务查询失败：${error.message}`, true); }
}
async function cancelTask() {
  if (!state.currentTaskId) return toast("尚无 task_id", true);
  try {
    const data = await requestJson(`/api/tasks/${state.currentTaskId}/cancel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: $("cancel-reason").value || "用户请求取消任务", requested_by: "web_console" }) });
    renderTask({ task: data.task, celery_state: data.revoke_requested ? "REVOKE_REQUESTED" : "REVOKE_UNAVAILABLE" });
  } catch (error) { toast(`取消失败：${error.message}`, true); }
}
async function listStaleTasks() {
  try { $("task-box").textContent = JSON.stringify(await requestJson("/api/task-recovery/stale"), null, 2); } catch (error) { toast(`失联任务查询失败：${error.message}`, true); }
}
async function recoverTask() {
  if (!state.currentTaskId) return toast("尚无 task_id", true);
  if (!$("worker-stopped").checked) return toast("恢复前必须确认旧 Worker 已停止", true);
  try {
    const key = `web-recovery-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const data = await requestJson(`/api/tasks/${state.currentTaskId}/recover`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: $("recovery-reason").value, requested_by: "web_console", confirmed_worker_stopped: true, idempotency_key: key }) });
    state.currentTaskId = data.recovery.task.task_id;
    renderTask({ task: data.recovery.task, celery_state: "PENDING" });
  } catch (error) { toast(`恢复失败：${error.message}`, true); }
}
function activateTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
}
function bindEvents() {
  $("refresh-health").addEventListener("click", refreshHealth);
  $("refresh-runs").addEventListener("click", loadRecentRuns);
  $("run-workflow").addEventListener("click", runWorkflow);
  $("run-offline-eval").addEventListener("click", runOfflineEvaluation);
  $("upload-csv").addEventListener("click", uploadCsv);
  $("use-knowledge").addEventListener("change", (event) => { $("knowledge-options").hidden = !event.target.checked; });
  $("search-knowledge").addEventListener("click", searchKnowledge);
  $("call-tool").addEventListener("click", callTool);
  $("enqueue-task").addEventListener("click", enqueueTask);
  $("refresh-task").addEventListener("click", refreshTask);
  $("cancel-task").addEventListener("click", cancelTask);
  $("list-stale").addEventListener("click", listStaleTasks);
  $("recover-task").addEventListener("click", recoverTask);
  $("behavior-agent-filter").addEventListener("change", () => renderAgentBehaviors());
  $("behavior-type-filter").addEventListener("change", () => renderAgentBehaviors());
  $("behavior-status-filter").addEventListener("change", () => renderAgentBehaviors());
  $("export-agent-audit").addEventListener("click", exportAgentAudit);
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  window.addEventListener("resize", () => window.requestAnimationFrame(() => renderPredictionChart(state.chartPoints)));
}
async function init() {
  bindEvents();
  resetAgentActivity();
  await Promise.all([refreshHealth(), loadRecentRuns(), loadTools(), loadDefaultDataset()]);
  const latest = document.querySelector('.recent-run[data-source="yield_domain_stategraph"]') || document.querySelector(".recent-run");
  if (latest?.dataset.runId) await loadRun(latest.dataset.runId);
}
init();
