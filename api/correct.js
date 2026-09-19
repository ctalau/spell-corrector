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
import { closeSync, openSync, readFileSync, readSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const MODEL_PATH = path.join(
  process.cwd(),
  "artifacts/spell_slm_m7_q4/qwen35_0_8b_distill_q4-Q4_K_M.gguf"
);
const PROMPT_TEMPLATE = readFileSync(
  path.join(process.cwd(), "artifacts/spell_slm_m7_q4/direct_correct_v1.txt"),
  "utf-8"
);
const CONTEXT_SIZE = 1024;
const MAX_ANSWER_TOKENS = 5;
const THREADS = Math.max(1, Math.min(4, os.cpus().length));

// Module-level singleton: persists across warm invocations of the same
// container, so the ~500MB model is loaded once per cold start, not once
// per request.
let modelState;

// The model is tracked by Git LFS. A host that clones without LFS ships the
// ~130-byte pointer file, whose first four bytes are `vers` -- node-llama-cpp
// then reports `Invalid GGUF magic. Expected "GGUF" but got "vers".`, which
// says nothing about what to fix. Check the magic ourselves and say it.
function assertRealModelFile() {
  let fd;
  try {
    fd = openSync(MODEL_PATH, "r");
  } catch {
    throw new Error(`model file missing at ${MODEL_PATH}`);
  }
  const head = Buffer.alloc(4);
  try {
    readSync(fd, head, 0, 4, 0);
  } finally {
    closeSync(fd);
  }
  const magic = head.toString("latin1");
  if (magic === "GGUF") return;
  if (magic === "vers") {
    throw new Error(
      "the deployed .gguf is a Git LFS pointer, not the model: enable Git LFS " +
        "for this project (Settings -> Git -> Git LFS) and redeploy"
    );
  }
  throw new Error(`unexpected magic ${JSON.stringify(magic)} in ${MODEL_PATH}, expected "GGUF"`);
}

async function getModelState() {
  if (modelState == null) {
    modelState = (async () => {
      const t0 = Date.now();
      assertRealModelFile();
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
  const userText = PROMPT_TEMPLATE.replace("{{SENTENCE}}", sentenceWithTypo);
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

function normalize(text) {
  let out = (text || "").trim();
  out = out.split(/\r?\n/)[0].trim();
  out = out.replace(/^["']+|["']+$/g, "").trim();
  out = out.split(/\s+/)[0] ?? "";
  return out.replace(/^["']+|["']+$/g, "").replace(/[.,;:!?]+$/g, "");
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
  if (typeof sentence !== "string" || !sentence.includes("<TYPO>") || !sentence.includes("</TYPO>")) {
    res.status(400).json({ error: 'body must be {"sentence": "..."} with exactly one <TYPO>word</TYPO> span' });
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
      correction: normalize(raw),
      raw,
      latency_ms: Date.now() - t0,
      cold_start_load_ms: loadMs,
    });
  } catch (err) {
    // A failed load poisons the singleton for the life of the container;
    // clear it so the next request retries instead of replaying the error.
    modelState = undefined;
    res.status(500).json({ error: String(err?.message ?? err) });
  }
}
