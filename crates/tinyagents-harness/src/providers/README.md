# providers

Model adapters whose behavior depends on TinyAgents prompt dialects rather
than on a plain hosted chat API. Both providers implement `ChatModel<()>`
(from `tinyinference_llm::model`) by shelling out to Anthropic's `claude` CLI
as a subprocess, but for different levels of agentic behavior:

| Module | CLI mode | Use when |
| --- | --- | --- |
| [`claude_agent_sdk`](claude_agent_sdk/README.md) | `claude -p` one-shot, no session | A single stateless completion is enough; no CLI-side tool use. |
| [`claude_code`](claude_code/README.md) | `claude -p --input-format stream-json --output-format stream-json`, session-resuming | The CLI should drive its own multi-step agentic loop (built-in file/shell tools, optional MCP) across a multi-turn conversation. |

Everywhere else in the harness a "provider" means a `tinyinference_llm`
adapter that speaks a vendor's native API directly (OpenAI, Anthropic
Messages, etc.); those live in `tinyinference_llm`, not here. This directory
exists specifically for the subset of providers whose wire format is a local
CLI subprocess rather than an HTTP API, and whose framing therefore has to
match TinyAgents' own prompt-tool conventions (see `crate::tool`) instead of
a provider-native tool-calling contract.

## File map

| File | Role |
| --- | --- |
| `mod.rs` | Re-exports the two provider modules. |
| `claude_agent_sdk/` | Prompt-guided, stateless `claude -p` adapter. |
| `claude_code/` | Session-resuming, agentic `claude` CLI adapter with its own permission/sandbox model. |

## Selecting a provider

Neither module registers itself automatically; a host application constructs
the provider it wants (`ClaudeAgentSdkProvider::new`/`for_model` or
`ClaudeCodeProvider::new`/`from_env`) and hands it to the harness like any
other `ChatModel` implementation. See each submodule's README for
construction details, environment variables, and operational constraints.
