# Claude Code + Ollama + Visual Studio Code (Windows) --- End-to-End Setup

This guide explains how to install Visual Studio Code, Ollama, and
Claude Code on Windows, connect Claude Code to an Ollama model, and use
it inside the VS Code integrated terminal.

> **Important distinction**
>
> -   **Claude Code** is Anthropic's coding agent and command-line tool.
> -   **Ollama** can provide local or Ollama-hosted models through an
>     Anthropic-compatible API.
> -   **VS Code** is the editor where you open your project and run
>     Claude Code in its integrated terminal.
>
> When Claude Code is connected to Ollama, the model inference is
> provided by Ollama---not by an Anthropic Claude model. You do not need
> an Anthropic subscription or Anthropic API key for this Ollama
> connection. Ollama Cloud models may require Ollama sign-in and may
> have their own account/plan requirements.

------------------------------------------------------------------------

## 1. What you will build

``` text
Windows PC
│
├── Visual Studio Code
│   └── Open your project folder
│       └── Integrated PowerShell terminal
│           └── Claude Code CLI
│               └── Ollama-compatible Anthropic API
│                   └── Ollama
│                       ├── Local model (example: gemma4:latest)
│                       └── Cloud model (example: gemma4:31b-cloud, if available)
│
└── Ollama API: http://localhost:11434
```

Claude Code can inspect your project, explain code, edit files, and run
terminal commands subject to its permission controls.

------------------------------------------------------------------------

## 2. Prerequisites

-   Windows 10 or Windows 11.
-   Internet access for downloading software.
-   A machine with enough RAM/VRAM for the local model you choose, or an
    Ollama Cloud account for cloud models.
-   PowerShell.
-   A VS Code project folder to work with.

Open PowerShell and check whether the tools are already installed:

``` powershell
code --version
ollama --version
claude --version
```

If a command is not recognized, install that application using the steps
below, then reopen PowerShell.

------------------------------------------------------------------------

## 3. Install Visual Studio Code

### 3.1 Download VS Code

Download the Windows installer:

https://code.visualstudio.com/

For most users, choose **Windows User Installer**.

Run the installer. During setup, enable these options if offered: - Add
to PATH. - Add "Open with Code" to file context menus. - Register Code
as an editor for supported file types.

### 3.2 Verify VS Code

Close and reopen PowerShell:

``` powershell
code --version
```

If `code` is still not recognized, open VS Code from the Start menu. In
the installer, ensure the PATH option is selected, or restart Windows
after installation.

### 3.3 Open a project

Create a sample project folder:

``` powershell
mkdir "$HOME\Desktop\claude-ollama-demo"
cd "$HOME\Desktop\claude-ollama-demo"
code .
```

VS Code should open the folder.

------------------------------------------------------------------------

## 4. Install Ollama

### 4.1 Download Ollama

Download Ollama for Windows:

https://ollama.com/download

Run the installer. Open the Ollama desktop application if it does not
start automatically.

Open a **new** PowerShell terminal.

### 4.2 Verify Ollama

``` powershell
ollama --version
ollama list
```

Example model list:

``` text
NAME                    SIZE
gemma4:latest           9.6 GB
nomic-embed-text:latest  274 MB
llama3.2:latest         2.0 GB
```

Your installed models may be different.

### 4.3 Download a model if required

For example:

``` powershell
ollama pull gemma4
```

Or:

``` powershell
ollama pull llama3.2
```

Choose a model that is suitable for coding and tool use. Larger models
usually need more memory. A model being available for ordinary chat does
not guarantee that it will perform equally well with agentic coding
tools.

### 4.4 Test the model

``` powershell
ollama run gemma4
```

Ask:

``` text
Explain the difference between a Python list and a tuple.
```

Exit Ollama's interactive prompt:

``` text
/bye
```

### 4.5 Check the Ollama API

Ollama normally runs in the background on Windows. Test its local API:

``` powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If model information is returned, the local API is reachable.

### 4.6 Optional: sign in for Ollama Cloud

If you intend to use an Ollama-hosted cloud model:

``` powershell
ollama signin
```

Complete the browser sign-in flow. Cloud model access requires internet
connectivity and may depend on account availability or plan.

------------------------------------------------------------------------

## 5. Install Claude Code

There are two ways to proceed.

### Method A --- Let Ollama launch and configure Claude Code (recommended for this guide)

Ollama's integration can install Claude Code if necessary and configure
the connection.

From PowerShell:

``` powershell
ollama launch claude
```

Follow the prompts: 1. Choose Claude Code if asked which integration to
launch. 2. Select an Ollama model that is available to your Ollama
setup. 3. Confirm the launch.

To specify a model:

``` powershell
ollama launch claude --model gemma4:latest
```

Use the exact model tag shown by:

``` powershell
ollama list
```

If using an Ollama Cloud model, first sign in and confirm that the model
is available:

``` powershell
ollama signin
ollama list
```

Then use the exact cloud model tag shown by Ollama, for example:

``` powershell
ollama launch claude --model gemma4:31b-cloud
```

Do not assume that `gemma4:latest` and `gemma4:31b-cloud` are the same
model. They are separate model references.

### Method B --- Install Claude Code separately

If you want to install Claude Code yourself, open PowerShell and run the
official Windows installer:

``` powershell
irm https://claude.ai/install.ps1 | iex
```

Close and reopen PowerShell, then verify:

``` powershell
claude --version
claude doctor
```

This installs the Claude Code CLI. To route it through Ollama, configure
the environment variables in the next section.

> **Authentication note:** Running Claude Code without Ollama normally
> uses Anthropic authentication and billing. When using Ollama, make
> sure the Ollama endpoint and model are configured as described below.
> Do not complete an Anthropic login just to use your Ollama model.

------------------------------------------------------------------------

## 6. Connect Claude Code to Ollama manually (PowerShell)

Use this method if you installed Claude Code separately or need to
troubleshoot the provider connection.

### 6.1 Set the environment variables for the current PowerShell window

``` powershell
$env:ANTHROPIC_AUTH_TOKEN = "ollama"
$env:ANTHROPIC_API_KEY = ""
$env:ANTHROPIC_BASE_URL = "http://localhost:11434"
```

These variables apply to the current PowerShell process and programs
launched from it. If you close the terminal, set them again in the new
terminal.

### 6.2 Launch Claude Code with the exact Ollama model name

For example:

``` powershell
claude --model gemma4:latest
```

If you are using a cloud model that Ollama exposes and supports:

``` powershell
claude --model gemma4:31b-cloud
```

Only use a model identifier that is available through your Ollama setup.

### 6.3 Verify the connection

Inside Claude Code, run:

``` text
/status
```

Check the model/provider information.

In a second PowerShell window, run:

``` powershell
ollama ps
```

For a local model, the model should appear while Claude Code is actively
generating a response. Cloud models may not appear in the same way as
locally loaded models.

You can also test the API independently:

``` powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

**Important:** The fact that Claude Code opens successfully does not by
itself prove that it is using Ollama. Verify the provider and model, and
check Ollama activity.

------------------------------------------------------------------------

## 7. Use Claude Code inside VS Code

The simplest and most reliable workflow for this Ollama setup is to run
Claude Code from VS Code's integrated terminal.

### 7.1 Open your project in VS Code

From PowerShell:

``` powershell
cd "$HOME\Desktop\claude-ollama-demo"
code .
```

Or open VS Code manually and choose:

**File → Open Folder**

Select your project folder.

### 7.2 Open the integrated terminal

In VS Code:

-   Select **Terminal → New Terminal**, or
-   Press \`Ctrl + Shift + \`\` (backtick).

Make sure the terminal is PowerShell and its current directory is your
project folder.

### 7.3 Start Claude Code with Ollama

If you are using Ollama's launcher:

``` powershell
ollama launch claude --model gemma4:latest
```

If you installed Claude Code separately and are using the manual
environment-variable method, set the variables in this integrated
terminal:

``` powershell
$env:ANTHROPIC_AUTH_TOKEN = "ollama"
$env:ANTHROPIC_API_KEY = ""
$env:ANTHROPIC_BASE_URL = "http://localhost:11434"

claude --model gemma4:latest
```

Claude Code now starts in the VS Code terminal with the project folder
as its working directory.

### 7.4 Try these prompts

Start with read-only tasks:

``` text
Explain the folder structure of this project.
```

``` text
Find the main entry point and explain how the application starts.
```

``` text
Review this project and identify possible bugs. Do not change any files yet.
```

Then try a small change:

``` text
Create a Python file named hello.py that prints Hello from Claude Code and explain the code.
```

Review the proposed changes and approve actions when appropriate.

### 7.5 Useful VS Code shortcuts

  Action                     Shortcut
  -------------------------- ---------------------------
  Open integrated terminal   `Ctrl + Shift + backtick`
  Open command palette       `Ctrl + Shift + P`
  Open file                  `Ctrl + P`
  Save file                  `Ctrl + S`
  Open Explorer              `Ctrl + Shift + E`
  Toggle terminal            `Ctrl + backtick`

------------------------------------------------------------------------

## 8. Optional: VS Code Claude Code extension

Anthropic also provides a Claude Code extension for VS Code. It provides
a native editor experience rather than only a terminal interface.

To install: 1. Open VS Code. 2. Open Extensions using
`Ctrl + Shift + X`. 3. Search for **Claude Code**. 4. Verify that the
publisher is Anthropic. 5. Install the extension. 6. Follow its setup
instructions.

**Provider compatibility warning:** The native extension may use its own
authentication and provider configuration. Do not assume that
environment variables set in a PowerShell terminal are automatically
used by the extension. The Ollama-supported, documented workflow in this
guide is to launch Claude Code from the integrated terminal using
`ollama launch claude` or the manual Ollama environment variables.

If the extension prompts you to sign in to Anthropic, that is a
different authentication path from the Ollama terminal integration. You
can continue using Claude Code in the integrated terminal without
installing the extension.

------------------------------------------------------------------------

## 9. Understanding the model warning

You may see a warning similar to:

``` text
Model "gemma4" isn't recognized
```

This can happen when the model identifier passed to Claude Code does not
match a model identifier it recognizes as a Claude model.

Remember: - `gemma4` is an Ollama model family/name, not an Anthropic
Claude model. - Ollama provides an Anthropic-compatible API so Claude
Code can communicate with compatible Ollama models. - Use the exact
Ollama tag, including any tag suffix, when launching. - A warning about
model recognition is not proof that the local model is working.

Check:

``` powershell
ollama list
openclaw --version
claude --version
```

For this guide, the relevant checks are `ollama list`,
`claude --version`, Claude Code's `/status`, and `ollama ps`. (The
`openclaw` command is unrelated and is not required.)

If the warning remains, update Ollama and Claude Code, then retry:

``` powershell
ollama --version
claude --version
claude doctor
```

Ollama's launcher is generally preferable to manually editing provider
settings because it configures the integration for you.

------------------------------------------------------------------------

## 10. Common daily workflow

### Start your coding session

1.  Make sure Ollama is running.
2.  Open your project in VS Code.
3.  Open the integrated PowerShell terminal.
4.  Start Claude Code through Ollama:

``` powershell
ollama launch claude --model gemma4:latest
```

5.  Ask Claude Code to inspect or work on your project.

### End your session

Exit Claude Code using:

``` text
/exit
```

You can close VS Code when finished. Ollama may continue running in the
background.

### Switch projects

Open another folder in VS Code, open a new integrated terminal, and
launch Claude Code from that folder.

------------------------------------------------------------------------

## 11. Troubleshooting

### Issue 1: `ollama` is not recognized

-   Confirm Ollama is installed.
-   Close and reopen PowerShell.
-   Run:

``` powershell
ollama --version
```

### Issue 2: `claude` is not recognized

Run the official installer:

``` powershell
irm https://claude.ai/install.ps1 | iex
```

Close and reopen PowerShell, then check:

``` powershell
claude --version
```

If you use `ollama launch claude`, Ollama may offer to install Claude
Code for you.

### Issue 3: Ollama API connection refused

Test:

``` powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

Open the Ollama desktop app and retry. Confirm that the local Ollama
server is running.

### Issue 4: Model not found

Check the exact model name:

``` powershell
ollama list
```

Use the same exact tag with Claude Code. For example, `gemma4:latest`
and `gemma4:31b-cloud` are not interchangeable.

### Issue 5: Claude Code opens but seems to use Anthropic billing

Check the launch method and provider configuration.

For a manual Ollama connection, set all three variables in the same
PowerShell terminal:

``` powershell
$env:ANTHROPIC_AUTH_TOKEN = "ollama"
$env:ANTHROPIC_API_KEY = ""
$env:ANTHROPIC_BASE_URL = "http://localhost:11434"

claude --model gemma4:latest
```

Then run `/status` inside Claude Code.

### Issue 6: Model is slow or runs out of memory

-   Try a smaller model.
-   Close memory-heavy applications.
-   Check GPU/RAM availability.
-   Use an Ollama Cloud model if you have access and prefer hosted
    inference.
-   Keep in mind that a coding agent may need a larger context window
    than a simple chat prompt.

### Issue 7: Claude Code cannot edit or run commands

Claude Code uses permission prompts for file edits and terminal
commands. Read the action carefully and approve only what you intend.

Do not use dangerous permission-bypass flags on your normal development
machine.

### Issue 8: VS Code terminal displays strange characters

Try:

``` text
/terminal-setup
```

Follow the instructions Claude Code provides, then reload VS Code if
needed.

------------------------------------------------------------------------

## 12. Security best practices

-   Do not paste API keys, passwords, or tokens into prompts.
-   Do not commit `.env`, credentials, or private keys to Git.
-   Review file changes before accepting them.
-   Review shell commands before approving them.
-   Do not use permission-bypass options on projects you do not trust.
-   Keep your Ollama API bound to localhost unless you have a secure
    network design.
-   Be careful when allowing an AI coding agent to read or modify
    sensitive project files.

------------------------------------------------------------------------

## 13. Final verification checklist

-   [ ] VS Code is installed.
-   [ ] `code --version` works.
-   [ ] Ollama is installed and running.
-   [ ] `ollama list` shows the intended model.
-   [ ] Ollama API responds at `http://localhost:11434/api/tags`.
-   [ ] Claude Code is installed.
-   [ ] `claude --version` works.
-   [ ] Claude Code is launched through Ollama.
-   [ ] Claude Code `/status` shows the intended provider/model
    configuration.
-   [ ] `ollama ps` shows local model activity when a local model is
    generating.
-   [ ] VS Code opens the intended project folder.
-   [ ] Claude Code can inspect the project and complete a small
    approved edit.

------------------------------------------------------------------------

## 14. Official documentation

-   Visual Studio Code download: https://code.visualstudio.com/
-   VS Code Windows installation:
    https://code.visualstudio.com/docs/setup/windows
-   Ollama download: https://ollama.com/download
-   Ollama documentation: https://docs.ollama.com/
-   Ollama Claude Code integration:
    https://docs.ollama.com/integrations/claude-code
-   Ollama VS Code integration:
    https://docs.ollama.com/integrations/vscode
-   Claude Code installation: https://code.claude.com/docs/en/setup
-   Claude Code getting started:
    https://code.claude.com/docs/en/getting-started
-   Claude Code VS Code integration:
    https://code.claude.com/docs/en/vs-code

------------------------------------------------------------------------

## 15. Quick command reference

``` powershell
# Check installations
code --version
ollama --version
claude --version

# Ollama
ollama list
ollama pull gemma4
ollama run gemma4
ollama ps
Invoke-RestMethod http://localhost:11434/api/tags

# Install Claude Code separately (optional)
irm https://claude.ai/install.ps1 | iex

# Launch Claude Code through Ollama
ollama launch claude
ollama launch claude --model gemma4:latest

# Manual Ollama provider setup (same PowerShell window)
$env:ANTHROPIC_AUTH_TOKEN = "ollama"
$env:ANTHROPIC_API_KEY = ""
$env:ANTHROPIC_BASE_URL = "http://localhost:11434"
claude --model gemma4:latest

# Open project in VS Code
cd "$HOME\Desktop\claude-ollama-demo"
code .

# Diagnostics
claude doctor
```

**End of guide.**
