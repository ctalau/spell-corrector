export const DEFAULT_ENDPOINT_ID = "835g1wte9tgcor";
export const DEFAULT_MODEL_NAME = "ctalau/qwen35-08b-spell-m7-distill";
export const RUNPOD_DEADLINE_MS = 55_000;
const POLL_MS = 1_500;

export function runpodConfig(env = process.env) {
  const apiKey = env.RUNPOD_API_KEY || "";
  const endpointId = env.RUNPOD_ENDPOINT_ID || DEFAULT_ENDPOINT_ID;
  return {
    apiKey,
    endpointId,
    configured: Boolean(apiKey),
  };
}

export function buildRunpodInput(userText) {
  return {
    openai_route: "/v1/chat/completions",
    openai_input: {
      model: DEFAULT_MODEL_NAME,
      messages: [{ role: "user", content: userText }],
      temperature: 0,
      max_tokens: 5,
      chat_template_kwargs: { enable_thinking: false },
    },
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function remainingMs(startedAt, deadlineMs) {
  return Math.max(1_000, deadlineMs - (Date.now() - startedAt));
}

async function runpodFetch(path, { apiKey, endpointId, method, body, timeoutMs }) {
  const url = `https://api.runpod.ai/v2/${endpointId}${path}`;
  const res = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: body == null ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    const err = new Error(`RunPod returned HTTP ${res.status} with a non-JSON body`);
    err.statusCode = res.status >= 400 ? res.status : 502;
    throw err;
  }
  if (!res.ok) {
    const detail = data.error?.message || data.error || data.status || text.slice(0, 200);
    const err = new Error(`RunPod returned HTTP ${res.status}: ${detail}`);
    err.statusCode = res.status >= 400 && res.status < 500 ? res.status : 502;
    err.details = { status: data.status, job_id: data.id };
    throw err;
  }
  return data;
}

function pending(status) {
  return status === "IN_QUEUE" || status === "IN_PROGRESS";
}

export async function runpodRunsync(userText, { apiKey, endpointId, deadlineMs = RUNPOD_DEADLINE_MS }) {
  const startedAt = Date.now();
  let job = await runpodFetch("/runsync", {
    apiKey,
    endpointId,
    method: "POST",
    body: { input: buildRunpodInput(userText) },
    timeoutMs: remainingMs(startedAt, deadlineMs),
  });

  while (pending(job.status) && job.id) {
    if (Date.now() - startedAt > deadlineMs - POLL_MS) {
      const err = new Error(
        "RunPod worker is still starting (cold start). Wait a few seconds and retry."
      );
      err.statusCode = 504;
      err.details = { status: job.status, job_id: job.id };
      throw err;
    }
    await sleep(POLL_MS);
    job = await runpodFetch(`/status/${encodeURIComponent(job.id)}`, {
      apiKey,
      endpointId,
      method: "GET",
      timeoutMs: remainingMs(startedAt, deadlineMs),
    });
  }

  if (job.status && job.status !== "COMPLETED") {
    const err = new Error(job.error || `RunPod job ${job.status}`);
    err.statusCode = 502;
    err.details = { status: job.status, job_id: job.id };
    throw err;
  }

  return job;
}
