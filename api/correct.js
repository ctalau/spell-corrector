// Vercel serverless function serving the M7 Q4_K_M distilled spelling
// corrector (artifacts/spell_slm_m7_q4) over HTTP.
//
// Uses node-llama-cpp rather than a spawned llama-server binary: its prebuilt
// native bindings are fetched for the correct target during `vercel build`,
// so there's no glibc/ABI mismatch between a locally-compiled binary and
// Vercel's Lambda-based runtime.
//
// IMPORTANT: the prompt is built with `SpecialTokensText`, not a plain JS
// string. Passing a plain string to node-llama-cpp tokenizes it with
// `special: false`, which turns the chat delimiters (`<|im_start|>`,
// `<|im_end|>`) into literal text instead of the model's actual special
// tokens -- verified against this exact model to cost ~6 accuracy points
// (86.0% -> 80.7% casefold on a 300-example slice of the recorded test set)
// versus the llama-server reference the M7 README's numbers came from.
import { getLlama, LlamaCompletion, LlamaText, SpecialTokensText } from "node-llama-cpp";
import os from "node:os";
import path from "node:path";
import { buildUserPrompt, MAX_ANSWER_TOKENS, normalizeCorrection, validateSentence } from "./lib/spelling.js";

const MODEL_PATH = path.join(
  process.cwd(),
  "artifacts/spell_slm_m7_q4/qwen35_0_8b_distill_q4-Q4_K_M.gguf"
);
const CONTEXT_SIZE = 1024;
const THREADS = Math.max(1, Math.min(4, os.cpus().length));

// Module-level singleton: persists across warm invocations of the same
// container, so the ~500MB model is loaded once per cold start, not once
// per request.
let modelState;
async function getModelState() {
  if (modelState == null) {
    modelState = (async () => {
      const t0 = Date.now();
      const llama = await getLlama({ gpu: false });
      const model = await llama.loadModel({ modelPath: MODEL_PATH });
      const context = await model.createContext({ contextSize: CONTEXT_SIZE, threads: THREADS });
      const sequence = context.getSequence();
      const completion = new LlamaCompletion({ contextSequence: sequence });
      return { completion, loadMs: Date.now() - t0 };
    })();
  }
  return modelState;
}

function buildPrompt(sentenceWithTypo) {
  const userText = buildUserPrompt(sentenceWithTypo);
  return LlamaText([
    new SpecialTokensText("<|im_start|>user\n"),
    userText,
    // The trailing `<think>\n\n</think>\n\n` is Qwen3.5's jinja template
    // output for `enable_thinking=False` (a.k.a. llama.cpp's
    // `--reasoning off`) -- extracted directly from the GGUF's own
    // `tokenizer.chat_template` metadata rather than guessed, since with
    // reasoning left on the whole `max_tokens` budget goes to an opened
    // <think> block and every answer comes back empty.
    new SpecialTokensText("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
  ]);
}

export const config = { maxDuration: 60 };

export default async function handler(req, res) {
  if (req.method === "GET") {
    res.status(200).json({
      status: "ok",
      model: "qwen35_0_8b_distill_q4-Q4_K_M",
      usage: 'POST { "sentence": "... <TYPO>word</TYPO> ..." }',
    });
    return;
  }
  if (req.method !== "POST") {
    res.status(405).json({ error: "use GET for status or POST for a correction" });
    return;
  }

  const sentence = req.body?.sentence;
  const invalid = validateSentence(sentence);
  if (invalid) {
    res.status(400).json({ error: invalid });
    return;
  }

  try {
    const { completion, loadMs } = await getModelState();
    const t0 = Date.now();
    const raw = await completion.generateCompletion(buildPrompt(sentence), {
      maxTokens: MAX_ANSWER_TOKENS,
      temperature: 0,
      repeatPenalty: false,
    });
    res.status(200).json({
      correction: normalizeCorrection(raw),
      raw,
      latency_ms: Date.now() - t0,
      cold_start_load_ms: loadMs,
    });
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
}
