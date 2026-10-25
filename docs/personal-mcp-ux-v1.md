# Personal MCP mobile UX prototype v1

Status: approved for stage 0. Contract: personal-mcp.v1.

Scenario goal: the user pastes a fragment from MCP documentation and, before the
first launch, understands what exactly will be installed, which resources will be
reserved and which secrets will be required. The basic flow has no Docker image,
HTTP port, shell command or local Node/npm installation.

## 1. Accepted launch decision

**Assigning an MCP to a room does not start it automatically.**

After a successful check, the "Add to rooms" button creates a binding only for an
MCP in the "Verified" status. If the MCP is stopped, the binding is kept, but its
tools are temporarily unavailable to the agents of that room. To restore access,
the user explicitly presses "Start and verify"; before the action the UI shows the
count, the reserved RAM and the remaining budget.

Reasons for the decision:

- assignment on its own must not unexpectedly consume RAM/CPU and monthly
  hours;
- stopping must be a safe way to free the running/RAM budget;
- the user always sees a separate lifecycle action and can understand why an
  MCP is not visible to the agent;
- a future opt-in autostart will need a separate warning and a feature flag.

## 2. Mobile flow

~~~mermaid
flowchart TD
  A["Personal MCP → Add"] --> B["From configuration"]
  B --> C["Paste mcpServers JSON"]
  C --> D["Candidate preview"]
  D --> E{"Secrets needed?"}
  E -- "Yes" --> F["Pick a Legion Secret per env name"]
  E -- "No" --> G["Pick RAM/storage preset"]
  F --> G
  G --> H["Connect and verify"]
  H --> I["Progress: package → runner → MCP probe"]
  I --> J{"Success?"}
  J -- "No" --> K["Safe error + retry/remedy"]
  J -- "Yes" --> L["Verified: tools and pinned version"]
  L --> M["Add to rooms"]
  M --> N["Room sees tools"]
~~~

On a narrow screen each step takes one column and needs no horizontal
scrolling. The continue button stays at the bottom of the viewport, and the JSON
preview and technical details are collapsed.

## 3. Screens and copy

### 3.1 Entry: "Add MCP"

Title: "Add MCP server"

Primary card:

- Label: "From configuration"
- Description: "Recommended for npx and npm exec commands from MCP documentation."
- Textarea label: "Paste the mcpServers JSON from the documentation"
- Placeholder:
  {"mcpServers":{"depwire":{"command":"npx","args":["-y","depwire-cli","mcp"]}}}
- Secondary link: "Or enter a single command"
- Help text: "Legion installs the package in an isolated cloud runner. Docker,
  npm install -g and local paths are not needed."

The existing paths stay below:

- "Marketplace" — a ready-made Legion server;
- "By HTTP(S) link" — an external HTTP MCP;
- "Advanced mode → Docker image" — for a published HTTP MCP image.

The managed flow does not show image, port, host, volume or shell fields.

### 3.2 Preview: "What will be started"

After debounce/submit, import-preview is called. While parsing is in progress:

"Checking the configuration…"

For each candidate:

- source name: "depwire";
- editable display name: "Depwire";
- badge: "Node.js · npx";
- package line: "Package depwire-cli";
- command line: "Command mcp";
- transport line: "stdio → internal HTTP bridge";
- pinning warning: "The package version will be pinned before launch.";
- resource presets: "512 MB", "1 GB", "2 GB" — only those available on the plan;
- storage presets: "512 MB", "1 GB", "5 GB" — only those available on the plan;
- checkbox: "Connect this server".

Copy for several servers:
"Servers found: 3. Choose which ones to add to Legion."

Copy for a rejected candidate:
"This server is not supported in basic mode. Legion accepts only npx -y/--yes or
npm exec --yes with a package from the npm registry. A GitHub link, local path,
shell command or Docker image is not run here. Choose "Advanced mode → Docker
image" or paste a supported block."

Technical link:
"Show normalized command" → expands argv with the note "Shell: not used".

### 3.3 Secrets: "Access to the service"

Title: "Choose secrets for the MCP"

Intro:
"Legion passes the selected secrets only to this MCP and only while it runs.
Values are not shown again and never end up in the configuration or logs."

For each env name:

- "DEPWIRE_API_KEY"
- select: "Choose a Legion secret"
- option: "No secret"
- link: "Create a secret"

If an env name is not bound:
"The server may fail to start without this secret. Choose a secret, or continue
without it if it is really not needed."

If the user pasted a plaintext env value:
"A secret value cannot be pasted into the MCP configuration. Save it in Legion
Secrets and choose a secret for the variable DEPWIRE_API_KEY."

After creation the list shows only:
"DEPWIRE_API_KEY · GitHub token · Connected"

Never show the value, a request body with the value, stdout or an exception with
the value.

### 3.4 Progress: "Connecting and verifying"

Title: "Connecting "Depwire""

Linear steps:

1. "Configuration accepted"
2. "Pinning the package version"
3. "Starting the isolated runner"
4. "Verifying the MCP"
5. "Fetching the tool list"

Helper text:
"This can take up to a minute. Don't close the screen — progress is kept if you
come back later."

After pinning:
"Version 1.4.2 pinned. Updating is a separate action."

The technical correlation id is shown only in an expandable block:
"Diagnostic code: 01JXXXXXVERIFY"

### 3.5 Ready: "Verified"

Status badge: "Verified"

Summary:
"Node.js · depwire-cli@1.4.2 · stdio bridge"

Tools:
"Tools: 2"
"search"
"fetch"

Actions:

- primary: "Add to rooms";
- secondary: "Start again";
- overflow: "Stop", "Update", "Delete".

Room assignment sheet:

- title: "Add "Depwire" to rooms";
- only the user's rooms are listed;
- helper: "Assigning to a room does not start the MCP automatically.";
- disabled state: "Start and verify the MCP first";
- success: "The tools are available to the agents of the selected room."

### 3.6 Resource summary and card

Above the list:
"Running 1 / 1 · RAM reserved 1 GB / 1 GB · in use 80 MB ·
storage 420 MB / 1 GB"

Stopped card:
"Stopped · 0 MB RAM · uses 420 MB storage"
"Data and bindings are kept."

Start confirmation:
"Start the MCP? 1 GB of RAM will be reserved and space will be used within the
running-servers limit."

Stop confirmation:
"Stop the MCP? RAM is freed, data and bindings are kept."

Delete confirmation:
"Delete the MCP and 420 MB of data? RAM is freed, while persistent data and
bindings are deleted. This action cannot be undone."

Quota denial:
"Cannot start "Depwire": 512 MB of reserved RAM is available, 1024 MB is
required. Stop another MCP or choose a smaller profile."

### 3.7 Safe errors

"Package not found"
"Check the npm package name and run the check again."

"Package registry is temporarily unavailable"
"We did not start the MCP. Try again later."

"This is not an MCP server"
"The process did not answer MCP initialize. Check the command arguments or pick
another package."

"The MCP did not respond in time"
"The server did not finish the check within the allotted time. Try again; if the
error repeats, open the diagnostics."

"Could not verify the MCP"
"The server is stopped, the tools are not available to agents. Try again"

In all cases the following is shown next to the error:

- "Retry", if the error is retryable;
- "Edit configuration", if the error relates to the command/package/secret;
- "Diagnostic code: <correlation_id>";
- an expandable technical part without secrets and without full stdout.

## 4. UI states

| API actual_status | UI label | Primary action | Room availability |
| --- | --- | --- | --- |
| preparing | Preparing | Cancel/close | No |
| starting/verifying | Verifying… | Hidden until the result | No |
| running | Starting | Hidden until verify | No |
| verified | Verified | Add to rooms | Yes |
| stopped | Stopped | Start and verify | No, bindings are kept |
| updating | Updating… | Hidden until the result | No for the candidate |
| error | Verification error | Retry | No |
| deleting | Deleting… | Hidden | No |

Do not use the word "container" as the only label in the basic flow: next to the
technical diagnostics the user is shown the plain "isolated runner".

## 5. Mobile acceptance checklist

- The large textarea is at least 8 lines, supports paste and shows a JSON error
  near the field.
- Candidates and secrets are vertical cards; do not use wide tables.
- No port/image fields in the managed flow.
- A resource preset shows "reserved", not only "limit".
- Buttons use verbs and state the consequences of stop/delete explicitly.
- Errors fit in one column and do not require reading raw JSON.
- npx is displayed as "npx", not as "Node command"; an explanation sits next to
  it: "The package will be installed by Legion inside an isolated runner".
- Unsupported input explains exactly what is allowed and offers the Docker
  advanced path; it never says "unknown error".
- A screen reader receives the status text, the progress step and the disabled
  reason.
- At 320 px width there is no horizontal scroll and no clipped primary actions.
