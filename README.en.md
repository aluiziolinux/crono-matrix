# Crono Matrix

### Your model. Your hardware. Your control.

[Português](README.md) · [Installation guide](docs/INSTALL.md) ·
[Capabilities](docs/CAPABILITIES.md) · [Test report](docs/UNIVERSAL_VALIDATION_2026-09-05.md)

**A local control panel for large GGUF models, built around real llama.cpp
execution, hardware-aware planning, and an offline-capable desktop.**

Loading a model is only part of the job. Context size, KV precision, GPU
offload, CPU experts and multimodal projectors compete for limited memory.
Crono Matrix brings these choices into one workflow: inspect the GGUF and
hardware, propose a configuration, show the command, then confirm the actual
server state after loading. You retain manual controls.

This is an **alpha** with current validation focused on **Linux/NVIDIA**.
Universal compatibility is a goal, not a claim that every architecture,
accelerator and operating system has been tested.

## Why try it?

- **One core, two interfaces:** CustomTkinter desktop or a local Web panel.
- **Large-model planning:** context, KV cache, GPU layers, CPU MoE experts,
  batch sizes and memory pressure are considered together.
- **Offline-capable:** once dependencies, build and weights are installed,
  local inference does not require internet. Downloads, discovery and web
  tools remain optional online features.
- **Multimodal awareness:** separate image/audio projector metadata and
  runtime confirmation. Keep MMProj on CPU when preserving VRAM is useful.
- **Agent integration:** publish the loaded model's effective context,
  modalities and reasoning information; an OpenCode adapter is included.
- **Discovery and evaluation:** Hugging Face radar/downloads and a 23-axis
  Alpha Eval suite with configurable sampling and reasoning controls.

## Start on Linux

Install Git, Python 3.11+ with venv and Tcl/Tk, a C/C++ toolchain, CMake,
build tools, and Node.js/npm. NVIDIA builds also need a working driver and
compatible CUDA Toolkit. Consult the [detailed guide](docs/INSTALL.md).

```bash
git clone https://github.com/aluiziolinux/crono-matrix.git
cd crono-matrix
./scripts/setup.sh
./scripts/bootstrap_llama_cpp.sh
./.venv/bin/python launch_model_gui.py
```

For the Web panel instead:

```bash
./.venv/bin/python launch_model_web.py
```

Open **http://127.0.0.1:7860**. Select the llama.cpp and model directories,
choose a GGUF, inspect its proposed configuration, and start the server.
For multimodal input, provide the matching MMProj and enable it explicitly.
The llama-server API/WebUI uses its own configured port, normally 8080.

The bootstrap uses a pinned upstream revision plus tracked patches. It
selects CUDA when NVIDIA tools are present, or CPU otherwise. Build only
for CPU with `--cpu`; add Chromium for the native browser tool with
`--with-browser`. Limit build memory pressure with `CRONO_BUILD_JOBS=4`.
An existing complete checkout/build can also be selected in the interface.
Do not overwrite your patched llama.cpp checkout to upgrade.

## Evidence, not speed promises

On an RTX 3060 12 GiB / 31.2 GiB RAM system, the local validation recorded:

| Check | Observed result |
| --- | --- |
| GGUF metadata reads | 0.409 s for 4Beasts Q6_K; 0.268 s for Nemotron Omni Q4_K_M |
| Nemotron Omni runtime | 262,144 context tokens, 816 MiB CUDA KV, MMProj on CPU |
| Short text/image/audio tests | Correct arithmetic, shape/color identification and synthetic English audio transcription |
| Product test suite, September 5 | 105 automated tests without failures |

These are specific smoke tests, not proof of complex coding quality or
full-window long-context performance. Video was advertised by the runtime
but not tested with a clip in this round. Read the
[conditions and limitations](docs/UNIVERSAL_VALIDATION_2026-09-05.md).

## Connect an agent

Enable universal mode, wait for the server to become ready, and inspect
`.crono-agent/agent-local.json`. It describes the actual runtime capabilities.
Clients supporting environment-based configuration can use:

```bash
source .crono-agent/agent-local.env.sh
```

Other clients may need manual provider configuration. Publishing a capability
does not add support to a harness that lacks the relevant input format.
Audio output is not promised simply because a vocoder file is present.

## Safety and scope

Keep the launcher and API on loopback. Native shell, file and browser tools
can perform real actions: scope permissions and working directories carefully.
Swap is not a substitute for RAM bandwidth and cannot guarantee freedom from
OOM or stalls. Linux-specific memory controls need appropriate permissions.

No model weights, projectors, builds, private conversations or private Crono
MCP server are bundled. Native llama-server tools are separate from that MCP.
Windows/macOS and non-NVIDIA accelerators remain unverified for this product.

## Help shape the next improvement

Try your own workload and [report your results](https://github.com/aluiziolinux/crono-matrix/issues).
Include hardware, model/quantization, build, sanitized command, effective
context and measurements. Do not upload weights or private data.

```bash
make test
make release-check
```

If Crono Matrix helps your work, consider starring the repository, sharing a
reproducible test, or [supporting independent development](docs/SUPPORT.md).
Donations are optional and do not unlock features.

[Apache License 2.0](LICENSE). Third-party components and models retain their
own licenses; see [NOTICE.md](NOTICE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
