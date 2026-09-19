// Shared spelling-prompt helpers for the Vercel serving layer.
// The prompt text lives in the M7 artifact so the web functions stay
// byte-identical to the llama.cpp / vLLM evals.
import { readFileSync } from "node:fs";
import path from "node:path";

export const MAX_ANSWER_TOKENS = 5;

const PROMPT_PATH = path.join(
  process.cwd(),
  "artifacts/spell_slm_m7_q4/direct_correct_v1.txt"
);

let cachedTemplate;
export function loadPromptTemplate() {
  if (cachedTemplate == null) {
    cachedTemplate = readFileSync(PROMPT_PATH, "utf-8");
  }
  return cachedTemplate;
}

export function buildUserPrompt(sentenceWithTypo) {
  return loadPromptTemplate().replace("{{SENTENCE}}", sentenceWithTypo);
}

export function validateSentence(sentence) {
  if (typeof sentence !== "string" || !sentence.includes("<TYPO>") || !sentence.includes("</TYPO>")) {
    return 'body must be {"sentence": "..."} with exactly one <TYPO>word</TYPO> span';
  }
  return null;
}

// Same first-token cleanup the local GGUF handler has always used.
export function normalizeCorrection(text) {
  let out = (text || "").trim();
  out = out.split(/\r?\n/)[0].trim();
  out = out.replace(/^["']+|["']+$/g, "").trim();
  out = out.split(/\s+/)[0] ?? "";
  return out.replace(/^["']+|["']+$/g, "").replace(/[.,;:!?]+$/g, "");
}

function firstChoice(output) {
  if (output == null) return null;
  if (Array.isArray(output)) {
    for (const item of output) {
      const found = firstChoice(item);
      if (found) return found;
    }
    return null;
  }
  if (Array.isArray(output.choices) && output.choices[0]) {
    return output.choices[0];
  }
  if (output.output != null && output.output !== output) {
    return firstChoice(output.output);
  }
  return null;
}

function asText(value) {
  if (value == null) return "";
  if (Array.isArray(value)) {
    return value
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && typeof part.text === "string") return part.text;
        return "";
      })
      .join("");
  }
  if (typeof value === "string") return value;
  if (typeof value === "object" && typeof value.text === "string") return value.text;
  return "";
}

// Strip a leaked Qwen3.5 think block if enable_thinking did not take.
export function stripThinkBlocks(text) {
  return String(text || "")
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .trim();
}

export function extractRunpodError(job) {
  const output = job?.output ?? job;
  const chunks = Array.isArray(output) ? output : [output];
  for (const chunk of chunks) {
    if (chunk == null) continue;
    if (typeof chunk.error === "string" && chunk.error) return chunk.error;
    if (chunk.error && typeof chunk.error === "object") {
      const message = chunk.error.message || chunk.error.type;
      if (message) return String(message);
    }
  }
  if (typeof job?.error === "string" && job.error) return job.error;
  return null;
}

export function extractRunpodText(job) {
  const output = job?.output ?? job;
  const choice = firstChoice(output) || {};
  const message = choice.message || choice.delta || {};
  const raw =
    asText(message.content) ||
    asText(choice.text) ||
    asText(choice.tokens) ||
    asText(output?.text) ||
    asText(message.reasoning_content) ||
    asText(message.reasoning);
  return stripThinkBlocks(raw);
}
