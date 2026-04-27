# MCP Server

SML ships an [MCP](https://modelcontextprotocol.io/) server so that an LLM client (Claude Desktop, Cursor, …) can list, launch, monitor, and cancel SML jobs as native tools.

The server is built with [FastMCP](https://github.com/jlowin/fastmcp) and exposed at `swiss_ai_model_launch.mcp:mcp`.

## Available tools

When connected, the client sees tools roughly equivalent to:

- `list_systems()` — discover HPC targets
- `establish(system, partition, …)` — set a default system/partition for the session
- `list_preconfigured_models()` — browse the model catalog
- `launch_preconfigured_model(...)` — submit a job and stream its lifecycle
- `get_job_status(job_id)`
- `get_job_logs(job_id)`
- `cancel_job(job_id)`

The server reuses the same config (`~/.sml/config.yml`) and credentials as the CLI — run `sml init` first.

## Hooking it up to Claude Desktop

Add an entry to `claude_desktop_config.json` (typically `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "sml": {
      "command": "fastmcp",
      "args": ["run", "swiss_ai_model_launch.mcp:mcp"],
      "env": {
        "SML_FIRECREST_SYSTEM": "clariden",
        "SML_PARTITION": "normal"
      }
    }
  }
}
```

If `fastmcp` isn't on your `$PATH`, point at the binary in your virtualenv (e.g. `/path/to/.venv/bin/fastmcp`) or use `uv`:

```json
{
  "mcpServers": {
    "sml": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "swiss_ai_model_launch.mcp:mcp"],
      "cwd": "/absolute/path/to/model-launch"
    }
  }
}
```

Restart Claude Desktop. The `sml` tools should appear in the tool picker.

## Other MCP hosts

The same server works with any MCP-compatible client. Cursor, Continue, and others use a similarly-shaped JSON config — adapt the `command` / `args` block.

## Coming later

A Claude marketplace skill is the planned distribution. Until that lands, the JSON snippet above is the supported path.
