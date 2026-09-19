#!/usr/bin/env node
// Build-time guard for the Vercel deployment.
//
// `.gguf` is tracked by Git LFS (see .gitattributes). A plain `git clone`
// checks out a ~130-byte pointer file instead of the model, and the first
// four bytes of that pointer are `vers` (from "version https://git-lfs...").
// node-llama-cpp then fails at the first request with
//   Invalid GGUF magic. Expected "GGUF" but got "vers".
// which is a 500 from a deployment that built and shipped cleanly.
//
// Failing the build here turns that runtime mystery into a build error that
// says what to turn on.
import { openSync, readSync, closeSync, statSync } from "node:fs";
import path from "node:path";

const MODEL_PATH = path.join(
  process.cwd(),
  "artifacts/spell_slm_m7_q4/qwen35_0_8b_distill_q4-Q4_K_M.gguf"
);

function fail(message) {
  console.error(`\n[check_model_asset] ${message}\n`);
  process.exit(1);
}

let size;
try {
  size = statSync(MODEL_PATH).size;
} catch {
  fail(`model not found at ${MODEL_PATH}`);
}

const fd = openSync(MODEL_PATH, "r");
const head = Buffer.alloc(4);
readSync(fd, head, 0, 4, 0);
closeSync(fd);

const magic = head.toString("latin1");
if (magic === "GGUF") {
  console.log(`[check_model_asset] ok — GGUF magic, ${(size / 1024 ** 2).toFixed(0)} MiB`);
  process.exit(0);
}

if (magic === "vers") {
  fail(
    `${MODEL_PATH} is a Git LFS pointer (${size} bytes), not the model.\n` +
      `  On Vercel: Project Settings -> Git -> enable "Git LFS", then redeploy.\n` +
      `  Locally:   git lfs install && git lfs pull`
  );
}

fail(`${MODEL_PATH} has magic ${JSON.stringify(magic)}, expected "GGUF" (${size} bytes)`);
