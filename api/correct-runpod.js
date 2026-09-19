// Vercel proxy for the RunPod Serverless Flex vLLM endpoint serving
// ctalau/qwen35-08b-spell-m7-distill. The browser never sees RUNPOD_API_KEY.
import {
  buildUserPrompt,
  extractRunpodError,
  extractRunpodText,
  normalizeCorrection,
  validateSentence,
} from "./lib/spelling.js";
import { runpodConfig, runpodRunsync } from "./lib/runpod.js";

export const config = { maxDuration: 60 };

export default async function handler(req, res) {
  const { apiKey, endpointId, configured } = runpodConfig();

  if (req.method === "GET") {
    res.status(200).json({
      status: configured ? "ok" : "unconfigured",
      backend: "runpod",
      configured,
      endpoint_id: endpointId,
      model: "ctalau/qwen35-08b-spell-m7-distill",
      usage: 'POST { "sentence": "... <TYPO>word</TYPO> ..." }',
    });
    return;
  }
  if (req.method !== "POST") {
    res.status(405).json({ error: "use GET for status or POST for a correction" });
    return;
  }
  if (!configured) {
    res.status(503).json({
      error: "RUNPOD_API_KEY is not set on this deployment",
      backend: "runpod",
    });
    return;
  }

  const sentence = req.body?.sentence;
  const invalid = validateSentence(sentence);
  if (invalid) {
    res.status(400).json({ error: invalid });
    return;
  }

  try {
    const t0 = Date.now();
    const job = await runpodRunsync(buildUserPrompt(sentence), { apiKey, endpointId });
    const outputError = extractRunpodError(job);
    if (outputError) {
      res.status(502).json({
        error: outputError,
        backend: "runpod",
        job_id: job.id,
        status: job.status,
      });
      return;
    }
    const raw = extractRunpodText(job);
    res.status(200).json({
      correction: normalizeCorrection(raw),
      raw,
      latency_ms: Date.now() - t0,
      backend: "runpod",
      endpoint_id: endpointId,
      job_id: job.id,
      delay_ms: job.delayTime,
      execution_ms: job.executionTime,
    });
  } catch (err) {
    res.status(err.statusCode || 500).json({
      error: String(err?.message ?? err),
      backend: "runpod",
      ...(err.details || {}),
    });
  }
}
