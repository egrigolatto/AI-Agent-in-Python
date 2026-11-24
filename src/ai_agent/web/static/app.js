"use strict";

/*
 * Frontend vanilla del agente Web (decisión D-10).
 *
 * Contrato: el backend envía SSE cuyo `data` es JSON con un campo `type` que
 * replica el nombre de las dataclasses de eventos del núcleo
 * (`src/ai_agent/events.py`). Acá solo se "renderizan" eventos: toda la
 * lógica del agente vive en Python (RF-11).
 *
 * El stream se lee con fetch + ReadableStream (no EventSource) porque el
 * turno se lanza con POST y el prompt viaja en el cuerpo de la petición.
 */

let sessionId = null;
let currentAssistant = null; // burbuja en streaming; null = próxima fase
const pendingCards = {}; // tool_call_id → nodo de confirmación

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null && text !== undefined) node.textContent = text;
  return node;
}

function scrollBottom() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function addBubble(kind, text) {
  const node = el("div", `msg ${kind}`, text);
  $("chat").appendChild(node);
  scrollBottom();
  return node;
}

function addTool(text, isError = false) {
  $("chat").appendChild(el("div", `tool${isError ? " error" : ""}`, text));
  scrollBottom();
}

function setStatus(text, kind = "status") {
  const status = $("status");
  status.textContent = text;
  status.className = kind;
}

function setBusy(busy) {
  $("prompt").disabled = busy || !sessionId;
  $("send").disabled = busy || !sessionId;
}

function endAssistantPhase() {
  currentAssistant = null;
}

/* ---------- sesión ---------- */

async function createSession(event) {
  event.preventDefault();
  const body = {
    provider: $("provider").value,
    workdir: $("workdir").value.trim() || null,
  };
  if ($("model").value.trim()) body.model = $("model").value.trim();
  if ($("base-url").value.trim()) body.base_url = $("base-url").value.trim();

  setStatus("Conectando con el proveedor…");
  $("connect").disabled = true;
  try {
    const response = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    sessionId = data.session_id;
    $("chat").replaceChildren();
    addTool(
      `Sesión creada: ${data.provider} · ${data.model} · ${data.workdir} · ` +
      `herramientas: ${data.tools.join(", ")}`
    );
    setStatus(`Conectado: ${data.provider} · ${data.model} · ${data.workdir}`, "status ok");
    setBusy(false);
    $("prompt").focus();
  } catch (error) {
    setStatus(`No se pudo crear la sesión: ${error.message}`, "status error");
  } finally {
    $("connect").disabled = false;
  }
}

/* ---------- turno con stream SSE ---------- */

async function sendMessage(event) {
  event.preventDefault();
  const text = $("prompt").value.trim();
  if (!text || !sessionId) return;
  $("prompt").value = "";
  addBubble("user", text);
  setBusy(true);
  endAssistantPhase();
  try {
    const response = await fetch(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      addTool(`Error (${response.status}): ${error.detail || "respuesta inválida"}`, true);
    } else {
      await readSse(response, handleEvent);
    }
  } catch (error) {
    addTool(`Error de red: ${error.message}`, true);
  } finally {
    setBusy(false);
    endAssistantPhase();
    $("prompt").focus();
  }
}

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        try {
          onEvent(JSON.parse(line.slice(6)));
        } catch {
          /* evento corrupto: ignorar */
        }
      }
    }
  }
}

function handleEvent(ev) {
  switch (ev.type) {
    case "RunStarted":
      break;
    case "LLMResponseReceived":
      if (ev.prompt_tokens !== null && ev.prompt_tokens !== undefined) {
        addTool(
          `iteración ${ev.iteration} · ${ev.model}: ` +
          `${ev.prompt_tokens} prompt / ${ev.completion_tokens} respuesta tokens`
        );
      }
      break;
    case "MessageDelta":
      if (!currentAssistant) currentAssistant = addBubble("assistant", "");
      currentAssistant.textContent += ev.text;
      scrollBottom();
      break;
    case "ToolCallReceived": {
      endAssistantPhase();
      const args = ev.arguments ? JSON.stringify(ev.arguments) : ev.raw_arguments;
      const warning = ev.requires_confirmation ? " ⚠ esperando confirmación" : "";
      addTool(`🔧 ${ev.tool_name} · ${truncate(args, 240)}${warning}`, ev.is_error);
      break;
    }
    case "ConfirmationRequested":
      showConfirmation(ev);
      break;
    case "ConfirmationResolved":
      resolveConfirmationCard(ev);
      break;
    case "ToolExecuted":
      if (!ev.denied) {
        addTool(`${ev.is_error ? "✗" : "↳"} ${truncate(ev.result, 600)}`, ev.is_error);
      }
      break;
    case "FinalMessage":
      endAssistantPhase();
      addBubble("assistant", ev.content);
      break;
    case "MaxIterationsReached":
      addTool(`⚠ Se alcanzó el máximo de ${ev.max_iterations} iteraciones.`, true);
      break;
    case "turn_end":
      if (!ev.ok) addTool(`El turno terminó con error: ${ev.error}`, true);
      break;
  }
}

function truncate(text, max) {
  if (!text) return "";
  return text.length > max ? text.slice(0, max) + "…" : text;
}

/* ---------- confirmaciones human-in-the-loop (D-03) ---------- */

function showConfirmation(ev) {
  const card = el("div", "confirm");
  card.appendChild(el("div", "confirm-title", `El agente quiere ejecutar “${ev.tool_name}”`));
  card.appendChild(el("pre", "confirm-args", JSON.stringify(ev.arguments, null, 2)));
  const buttons = el("div", "confirm-buttons");
  const approve = el("button", "approve", "Aprobar");
  const deny = el("button", "deny", "Rechazar");
  approve.addEventListener("click", () => decide(ev.tool_call_id, true));
  deny.addEventListener("click", () => decide(ev.tool_call_id, false));
  buttons.append(approve, deny);
  card.appendChild(buttons);
  $("chat").appendChild(card);
  scrollBottom();
  pendingCards[ev.tool_call_id] = card;
}

async function decide(toolCallId, approved) {
  const card = pendingCards[toolCallId];
  if (!card) return;
  card.querySelectorAll("button").forEach((button) => (button.disabled = true));
  try {
    const response = await fetch(`/api/sessions/${sessionId}/confirmations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool_call_id: toolCallId, approved }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      card.appendChild(el("div", "confirm-error", `No se pudo enviar: ${error.detail || response.status}`));
      card.querySelectorAll("button").forEach((button) => (button.disabled = false));
    }
  } catch (error) {
    card.appendChild(el("div", "confirm-error", `Error de red: ${error.message}`));
    card.querySelectorAll("button").forEach((button) => (button.disabled = false));
  }
}

function resolveConfirmationCard(ev) {
  const card = pendingCards[ev.tool_call_id];
  if (!card) return;
  card.classList.add(ev.approved ? "approved" : "rejected");
  card.querySelectorAll("button").forEach((button) => (button.disabled = true));
  card.appendChild(
    el("div", "confirm-verdict", ev.approved ? "✓ Aprobada por el usuario" : "✗ Rechazada por el usuario")
  );
  delete pendingCards[ev.tool_call_id];
  scrollBottom();
}

/* ---------- arranque ---------- */

$("session-form").addEventListener("submit", createSession);
$("chat-form").addEventListener("submit", sendMessage);
