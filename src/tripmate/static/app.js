"use strict";

const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");
const loadingStatus = document.getElementById("loading-status");
const errorStatus = document.getElementById("error-status");
const CHAT_TIMEOUT_MS = 300000;
const TOOL_LABELS = {
  search_destination_guide: "Destination Guide",
  get_weather_forecast: "Weather Forecast",
};
const EVENT_LABELS = {
  agent_started: "Request started",
  tool_call: "Tool called",
  tool_result: "Tool completed",
  tool_error: "Tool failed",
  limit_reached: "Tool-call limit reached",
  agent_complete: "Response completed",
};
let busy = false;
const newChatButton = document.getElementById("new-chat");
const THREAD_KEY = "tripmate_thread_id";
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function saveThread(id) {
  try { localStorage.setItem(THREAD_KEY, id); } catch { /* Private storage may be disabled. */ }
  return id;
}
function currentThread() {
  try {
    const stored = localStorage.getItem(THREAD_KEY);
    if (UUID_PATTERN.test(stored || "")) return stored;
  } catch { /* Keep this page's conversation in memory. */ }
  return saveThread(crypto.randomUUID());
}
let threadId = currentThread();
newChatButton.addEventListener("click", () => {
  if (busy) return;
  threadId = saveThread(crypto.randomUUID());
  messages.replaceChildren();
  errorStatus.hidden = true;
  input.value = "";
  input.focus();
});

function setLoading(active) {
  busy = active;
  newChatButton.disabled = active;
  input.disabled = active;
  sendButton.disabled = active;
  sendButton.setAttribute("aria-busy", String(active));
  loadingStatus.hidden = !active;
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.disabled = active;
  });
}

function renderMessage(role, text) {
  document.getElementById("empty-state")?.remove();
  const article = document.createElement("article");
  article.className = `message message-${role}`;
  const label = document.createElement("h3");
  label.textContent = role === "user" ? "You" : "TripMate";
  const body = document.createElement("p");
  body.textContent = text;
  article.append(label, body);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function renderTrace(article, result) {
  const details = document.createElement("details");
  details.className = "execution-trace";
  const summary = document.createElement("summary");
  summary.textContent = "Execution trace";
  const tools = document.createElement("p");
  tools.textContent = result.tools_used.length
    ? `Tools used: ${result.tools_used.map((name) => TOOL_LABELS[name]).join(" → ")}`
    : "No tools were required for this response.";
  const list = document.createElement("ol");
  for (const event of result.trace) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = EVENT_LABELS[event.event]
      + (event.tool ? ` · ${TOOL_LABELS[event.tool]}` : "");
    item.append(title);
    // Only current, known safe tool arguments are displayed.
    const fields = event.tool === "search_destination_guide" ? ["query"]
      : event.tool === "get_weather_forecast" ? ["city", "date_or_month"] : [];
    for (const key of fields) {
      if (typeof event.arguments?.[key] === "string") {
        const argument = document.createElement("p");
        argument.textContent = `${key === "date_or_month" ? "period" : key}: ${event.arguments[key]}`;
        item.append(argument);
      }
    }
    if (event.message) {
      const note = document.createElement("p");
      note.textContent = event.message;
      item.append(note);
    }
    list.append(item);
  }
  details.append(summary, tools, list);
  article.append(details);
}

function validResponse(data) {
  return data && typeof data.answer === "string" && data.answer.trim()
    && ["completed", "limit_reached"].includes(data.status)
    && Array.isArray(data.tools_used) && data.tools_used.every((name) => Object.hasOwn(TOOL_LABELS, name))
    && Array.isArray(data.trace) && data.trace.every((event) => event
      && Object.hasOwn(EVENT_LABELS, event.event)
      && (event.tool == null || Object.hasOwn(TOOL_LABELS, event.tool))
      && (event.message == null || typeof event.message === "string"));
}

class TripMateError extends Error { }

function apiError(status, data = {}) {
  if (data.code === "rate_limited") {
    return Number.isInteger(data.retry_after_seconds) && data.retry_after_seconds >= 0
      ? `TripMate is temporarily rate limited. Try again in about ${data.retry_after_seconds} seconds.`
      : "TripMate is temporarily rate limited. Please try again shortly.";
  }
  if (typeof data.message === "string" && data.message) return data.message;
  if (status === 422) return "Enter a question between 1 and 4000 characters.";
  if (status === 503) return "TripMate AI service is currently unavailable. Please try again later.";
  if (status === 502) return "TripMate couldn't reach its AI service. Please try again.";
  return "TripMate couldn't complete that request. Please try again.";
}

async function sendMessage(event) {
  event.preventDefault();
  if (busy) return;
  const message = input.value.trim();
  errorStatus.hidden = true;
  if (!message || message.length > 4000) {
    errorStatus.textContent = "Enter a question between 1 and 4000 characters.";
    errorStatus.hidden = false;
    input.focus();
    return;
  }
  renderMessage("user", message);
  const article = renderMessage("assistant", "");
  const textNode = document.createTextNode("");
  article.querySelector("p").append(textNode);
  loadingStatus.textContent = "Connecting to TripMate…";
  setLoading(true);
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal: AbortSignal.timeout(CHAT_TIMEOUT_MS),
    });
    if (!response.ok) {
      let data = {};
      try { data = await response.json(); } catch { /* Use the status fallback. */ }
      throw new TripMateError(apiError(response.status, data));
    }
    if (!response.body) throw new Error("Streaming is unavailable in this browser.");
    const reader = response.body.getReader();
    let done = false;
    let started = false;
    const parser = createSSEParser((event) => {
      if (done || !event || typeof event.type !== "string") throw new Error("Invalid stream");
      if (event.type === "start") {
        if (started || event.thread_id !== threadId) throw new Error("Invalid stream");
        started = true;
      } else if (event.type === "error") {
        throw new TripMateError(apiError(0, event));
      } else if (!started) {
        throw new Error("Invalid stream");
      } else if (event.type === "token" && typeof event.content === "string") {
        textNode.appendData(event.content);
        loadingStatus.textContent = "TripMate is responding…";
      } else if (event.type === "tool_call" && Object.hasOwn(TOOL_LABELS, event.tool)) {
        loadingStatus.textContent = event.tool === "search_destination_guide"
          ? "Searching destination guide…" : "Checking weather…";
      } else if (event.type === "tool_result" && ["success", "error"].includes(event.status)) {
        loadingStatus.textContent = event.status === "error" ? "Tool unavailable; preparing a response…"
          : `${TOOL_LABELS[event.tool] || "Tool"} complete`;
      } else if (event.type === "done" && validResponse(event.result)
        && event.result.thread_id === threadId) {
        // Reconcile trimming or the locally generated tool-budget limitation.
        textNode.data = event.result.answer;
        renderTrace(article, event.result);
        done = true;
      } else {
        throw new Error("Invalid stream");
      }
      messages.scrollTop = messages.scrollHeight;
    });
    try {
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        parser.push(chunk.value);
      }
      parser.finish();
      if (!done) throw new Error("TripMate's response was interrupted. Please try again.");
    } finally {
      await reader.cancel();
      reader.releaseLock();
    }
    input.value = "";
  } catch (error) {
    errorStatus.textContent = error.name === "TimeoutError" || error.name === "AbortError"
      ? "This request took too long. Please try again."
      : error instanceof TypeError
        ? "Unable to connect to TripMate. Check your connection and try again."
        : error instanceof TripMateError
          ? error.message
          : "TripMate could not complete the request. Please try again.";
    const note = document.createElement("p");
    note.textContent = "Response incomplete.";
    article.append(note);
    errorStatus.hidden = false;
  } finally {
    setLoading(false);
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
}

form.addEventListener("submit", sendMessage);
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (busy) return;
    input.value = button.dataset.prompt;
    input.focus();
  });
});
setLoading(false);

async function checkHealth() {
  const status = document.getElementById("health-status");
  try {
    const response = await fetch("/api/health", { signal: AbortSignal.timeout(5000), cache: "no-store" });
    if (!response.ok || (await response.json()).status !== "ok") throw new Error("Unavailable");
    status.textContent = "Application online";
  } catch {
    status.textContent = "Application status unavailable · Refresh to retry";
  }
}
checkHealth();
