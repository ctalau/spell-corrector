import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildUserPrompt,
  extractRunpodError,
  extractRunpodText,
  normalizeCorrection,
  stripThinkBlocks,
  validateSentence,
} from "../api/lib/spelling.js";
import { buildRunpodInput, DEFAULT_ENDPOINT_ID, DEFAULT_MODEL_NAME, runpodConfig } from "../api/lib/runpod.js";
import runpodHandler from "../api/correct-runpod.js";

test("buildUserPrompt substitutes the marked sentence", () => {
  const prompt = buildUserPrompt("I <TYPO>crimbimg</TYPO> the hill.");
  assert.match(prompt, /I <TYPO>crimbimg<\/TYPO> the hill\./);
  assert.doesNotMatch(prompt, /\{\{SENTENCE\}\}/);
  assert.match(prompt, /ONLY the correctly spelled replacement word/);
});

test("validateSentence requires a TYPO span", () => {
  assert.equal(validateSentence("I <TYPO>teh</TYPO> cat"), null);
  assert.match(validateSentence("no typo here"), /TYPO/);
  assert.match(validateSentence(null), /TYPO/);
});

test("normalizeCorrection keeps the first cleaned token", () => {
  assert.equal(normalizeCorrection("  climbing\nextra"), "climbing");
  assert.equal(normalizeCorrection('"climbing."'), "climbing");
  assert.equal(normalizeCorrection("climbing the"), "climbing");
});

test("stripThinkBlocks drops leaked reasoning", () => {
  assert.equal(stripThinkBlocks("<think>\nnope\n</think>\n\nclimbing"), "climbing");
});

test("extractRunpodText reads OpenAI chat completions", () => {
  const job = {
    status: "COMPLETED",
    output: {
      choices: [{ message: { role: "assistant", content: "climbing" } }],
    },
  };
  assert.equal(extractRunpodText(job), "climbing");
});

test("extractRunpodText reads the live worker-vllm chat.completion wrapper", () => {
  const job = {
    delayTime: 272014,
    executionTime: 580,
    status: "COMPLETED",
    output: [
      {
        choices: [
          {
            finish_reason: "stop",
            message: { content: "climbing", role: "assistant" },
          },
        ],
        object: "chat.completion",
      },
    ],
  };
  assert.equal(extractRunpodText(job), "climbing");
  assert.equal(normalizeCorrection(extractRunpodText(job)), "climbing");
});

test("extractRunpodText handles streamed-style arrays and think tags", () => {
  const job = {
    output: [
      {
        choices: [
          {
            message: { content: "<think></think>\nclimbing" },
          },
        ],
      },
    ],
  };
  assert.equal(extractRunpodText(job), "climbing");
});

test("extractRunpodError reads worker error objects", () => {
  assert.equal(
    extractRunpodError({ output: { error: { message: "vLLM returned HTTP 400: bad" } } }),
    "vLLM returned HTTP 400: bad"
  );
  assert.equal(extractRunpodError({ output: { choices: [] } }), null);
});

test("buildRunpodInput is OpenAI chat with thinking off", () => {
  const input = buildRunpodInput("hello");
  assert.equal(input.openai_route, "/v1/chat/completions");
  assert.equal(input.openai_input.model, DEFAULT_MODEL_NAME);
  assert.equal(input.openai_input.temperature, 0);
  assert.equal(input.openai_input.max_tokens, 5);
  assert.deepEqual(input.openai_input.chat_template_kwargs, { enable_thinking: false });
  assert.deepEqual(input.openai_input.messages, [{ role: "user", content: "hello" }]);
});

test("runpodConfig reads env and defaults the endpoint id", () => {
  assert.deepEqual(runpodConfig({}), {
    apiKey: "",
    endpointId: DEFAULT_ENDPOINT_ID,
    configured: false,
  });
  assert.deepEqual(runpodConfig({ RUNPOD_API_KEY: "secret", RUNPOD_ENDPOINT_ID: "abc" }), {
    apiKey: "secret",
    endpointId: "abc",
    configured: true,
  });
});

function mockRes() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
  };
}

test("RunPod GET reports unconfigured without a key", async () => {
  const previous = process.env.RUNPOD_API_KEY;
  delete process.env.RUNPOD_API_KEY;
  try {
    const res = mockRes();
    await runpodHandler({ method: "GET" }, res);
    assert.equal(res.statusCode, 200);
    assert.equal(res.body.configured, false);
    assert.equal(res.body.backend, "runpod");
    assert.equal(res.body.endpoint_id, DEFAULT_ENDPOINT_ID);
  } finally {
    if (previous == null) delete process.env.RUNPOD_API_KEY;
    else process.env.RUNPOD_API_KEY = previous;
  }
});

test("RunPod POST without a key is 503", async () => {
  const previous = process.env.RUNPOD_API_KEY;
  delete process.env.RUNPOD_API_KEY;
  try {
    const res = mockRes();
    await runpodHandler({ method: "POST", body: { sentence: "I <TYPO>teh</TYPO> cat" } }, res);
    assert.equal(res.statusCode, 503);
    assert.match(res.body.error, /RUNPOD_API_KEY/);
  } finally {
    if (previous == null) delete process.env.RUNPOD_API_KEY;
    else process.env.RUNPOD_API_KEY = previous;
  }
});

test("RunPod POST rejects a sentence with no TYPO span", async () => {
  const previous = process.env.RUNPOD_API_KEY;
  process.env.RUNPOD_API_KEY = "test-key-not-used";
  try {
    const res = mockRes();
    await runpodHandler({ method: "POST", body: { sentence: "plain" } }, res);
    assert.equal(res.statusCode, 400);
    assert.match(res.body.error, /TYPO/);
  } finally {
    if (previous == null) delete process.env.RUNPOD_API_KEY;
    else process.env.RUNPOD_API_KEY = previous;
  }
});
