# Deploying the corrector to Vercel

`api/correct.js` loads `artifacts/spell_slm_m7_q4/qwen35_0_8b_distill_q4-Q4_K_M.gguf`
(505 MiB) from the deployment bundle with node-llama-cpp. Two Vercel project
settings are required for that to work; neither is expressible in `vercel.json`,
so a fresh project fails even though the repository is correct.

## 1. Git LFS (`Invalid GGUF magic. Expected "GGUF" but got "vers".`)

`.gguf` is tracked by Git LFS (`.gitattributes`). Unless Git LFS is enabled for
the project, Vercel clones without it and deploys the ~130-byte pointer file:

```
version https://git-lfs.github.com/spec/v1
oid sha256:4280225d...
size 529296832
```

`vers` is the first four bytes of that pointer — that is the "magic" the loader
reports. The build succeeds and the failure only shows up as a 500 on the first
request.

**Fix:** Project Settings → Git → enable **Git LFS**, then redeploy. The setting
is free on all plans and only takes effect on deployments made after it is on.

`scripts/check_model_asset.mjs` runs as the project's build command and fails
the build with this message if the pointer ever ships again.

## 2. Large functions (250 MB bundle limit)

With the real model in the bundle the function is ~505 MiB plus the
node-llama-cpp native binaries, over the standard 250 MB uncompressed limit.

**Fix:** set the environment variable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` on the
project (Settings → Environment Variables, or `vercel env add`). That opts into
the large-functions beta (up to 5 GB); it requires fluid compute with Active CPU,
which is the default for new projects. Not available with Secure Compute or
Static IPs.

## Expected performance

The 0.60 s p50 in `artifacts/spell_slm_m7_q4/README.md` was measured with 4
threads on a 4-vCPU box. Vercel functions get 1 vCPU on Hobby (2 vCPU max on
Pro), so per-request latency will be several times that, and each cold start
pays the ~500 MB model load on top.

## If neither option is available

Serve the model outside Vercel (a llama-server container on Runpod/Fly/Render)
and reduce `api/correct.js` to a proxy. The 4.5 MB request/response limit and
the function duration cap are not the binding constraints here; the bundle size
and 1 vCPU are.
