# OpenClaw + Ollama + Telegram (Windows) --- End-to-End Setup

A practical, repeatable guide for installing Ollama and OpenClaw on
Windows, connecting an Ollama model, opening the OpenClaw web dashboard,
and chatting with your assistant through Telegram.

> **Environment covered:** Windows 10/11, PowerShell, OpenClaw CLI,
> Ollama, and Telegram.
>
> **Your current setup (from the terminal screenshots):** - OpenClaw
> version shown: `2026.11.2` - Gateway port: `18789` - Ollama models
> previously listed: `gemma4:latest`, `nomic-embed-text:latest`, and
> `llama3.2:latest` - Existing OpenClaw config displayed:
> `ollama/gemma4:31b-cloud`
>
> **Important:** `gemma4:latest` and `gemma4:31b-cloud` are different
> model references. Check the models available to OpenClaw before
> changing your configured model. This guide shows how to verify and
> select either route.

------------------------------------------------------------------------

## 1. Architecture

``` text
Telegram user
     |
     v
Telegram Bot (created with @BotFather)
     |
     v
OpenClaw Gateway (Windows PC, port 18789)
     |
     +---------------------> OpenClaw Control UI (browser)
     |
     v
OpenClaw Agent
     |
     v
Ollama provider
     |
     +---- Local model (example: gemma4:latest)
     |
     +---- Ollama Cloud model (if signed in and available)
```

-   **Ollama** runs or provides access to the language model.
-   **OpenClaw** manages the agent, tools, sessions, and messaging
    connections.
-   **Gateway** is the service that connects the dashboard and Telegram
    to the agent.
-   **Telegram bot** is the chat interface on your phone.

The OpenClaw Gateway normally listens locally on port `18789`. Ollama's
local API normally listens on port `11434`.

------------------------------------------------------------------------

## 2. Prerequisites

You need:

1.  Windows 10 or Windows 11.
2.  Internet access for installation and Telegram.
3.  An Ollama model suitable for chat and tool use.
4.  A Telegram account.
5.  A modern browser.

Open PowerShell and check whether the commands are already installed:

``` powershell
ollama --version
openclaw --version
node --version
npm --version
```

If `ollama` or `openclaw` is not recognized, install the missing
application in the next sections.

------------------------------------------------------------------------

## 3. Install Ollama

### 3.1 Download and install

Download Ollama for Windows:

https://ollama.com/download

Run the installer and complete the installation.

Open a **new PowerShell window** after installation.

### 3.2 Verify Ollama

``` powershell
ollama --version
```

Check the installed models:

``` powershell
ollama list
```

Example:

``` text
NAME                    ID          SIZE
gemma4:latest           ...         9.6 GB
nomic-embed-text:latest ...         274 MB
llama3.2:latest         ...         2.0 GB
```

Your exact model list may be different.

### 3.3 Download a model if you need one

For example:

``` powershell
ollama pull gemma4
```

Or:

``` powershell
ollama pull llama3.2
```

Use a model that supports the tool-calling features you need. Model
quality, available RAM/VRAM, and context length affect performance.

### 3.4 Test the model directly

``` powershell
ollama run gemma4
```

Ask a simple question:

``` text
Explain what an API is in simple terms.
```

Exit the interactive Ollama session with:

``` text
/bye
```

### 3.5 Verify the Ollama API

Ollama normally runs as a background application on Windows. Check its
local API:

``` powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If this returns model information, the local Ollama API is reachable.

If it fails, open the Ollama desktop application or start the Ollama
service, then retry.

### 3.6 Optional: sign in to Ollama Cloud

If you want to use hosted Ollama models through your Ollama
installation, sign in:

``` powershell
ollama signin
```

Complete the sign-in flow in your browser.

Cloud models may require an internet connection and Ollama account
access. They are not the same as models downloaded and run fully
locally.

------------------------------------------------------------------------

## 4. Install OpenClaw

### 4.1 Install using the official PowerShell installer

Run:

``` powershell
iwr -useb https://openclaw.ai/install.ps1 | iex
```

The installer can install the required runtime and OpenClaw, then launch
onboarding.

If the installer completes but the command is not found, close and
reopen PowerShell, then check:

``` powershell
openclaw --version
```

### 4.2 Verify installation

``` powershell
openclaw --version
```

Check the configuration and environment:

``` powershell
openclaw doctor
```

### 4.3 Run onboarding if this is a fresh installation

``` powershell
openclaw onboard
```

Follow the prompts. Select Ollama when choosing the model provider, then
choose the appropriate connection mode:

-   **Local** --- use models served by the Ollama process on this
    computer.
-   **Cloud + Local** --- use local models and Ollama-hosted cloud
    models through a signed-in Ollama host.

If you already completed onboarding and OpenClaw is working, do not
repeat onboarding unnecessarily. Use the configuration commands in the
next section.

------------------------------------------------------------------------

## 5. Connect OpenClaw to Ollama

### 5.1 Inspect the models OpenClaw can see

``` powershell
openclaw models list --provider ollama
```

Also check Ollama itself:

``` powershell
ollama list
```

These commands answer different questions: - `ollama list` shows models
available from your Ollama installation. -
`openclaw models list --provider ollama` shows models OpenClaw knows
about for that provider.

### 5.2 Select a local model

If `gemma4:latest` appears in the OpenClaw model list, select it:

``` powershell
openclaw models set ollama/gemma4:latest
```

You can select another model if it is listed, for example:

``` powershell
openclaw models set ollama/llama3.2:latest
```

Use the exact provider/model identifier shown by your OpenClaw model
list.

### 5.3 Use your existing cloud model instead

Your existing OpenClaw configuration previously displayed:

``` text
ollama/gemma4:31b-cloud
```

That is a cloud model reference, not the same as `gemma4:latest`.

To use it: 1. Sign in to Ollama using `ollama signin`. 2. Confirm that
the cloud model is available to your Ollama account. 3. Check whether
OpenClaw lists it:

``` powershell
openclaw models list --provider ollama
```

4.  If it is listed, select the exact model identifier:

``` powershell
openclaw models set ollama/gemma4:31b-cloud
```

Do not run this command if that exact model is not listed. Choose a
model that is actually available.

### 5.4 Verify the selected model

``` powershell
openclaw models list
```

Then open the dashboard and send a simple test prompt.

For a local model, you can also check whether Ollama has loaded it:

``` powershell
ollama ps
```

------------------------------------------------------------------------

## 6. Start the OpenClaw Gateway

The Gateway connects the model, agent, dashboard, and messaging
channels.

### Option A --- Run in the foreground (useful for learning and debugging)

``` powershell
openclaw gateway
```

Keep this terminal open. Closing it or pressing `Ctrl+C` stops the
foreground Gateway.

### Option B --- Install and run it as a background service

If you want the Gateway to keep running in the background:

``` powershell
openclaw gateway install
openclaw gateway start
```

Check its status:

``` powershell
openclaw gateway status
```

If you change configuration and need to restart:

``` powershell
openclaw gateway restart
```

Use either the foreground Gateway or the background service. Avoid
starting duplicate Gateway processes.

------------------------------------------------------------------------

## 7. Open the OpenClaw Web UI

With the Gateway running, execute:

``` powershell
openclaw dashboard
```

This should open the Control UI in your browser.

You can also open it manually:

http://127.0.0.1:18789/

If the page does not load, check:

``` powershell
openclaw gateway status
```

The dashboard is local to your computer. Do not expose the Gateway port
directly to the public internet.

------------------------------------------------------------------------

## 8. Enable web search and web fetch (optional)

OpenClaw's web search and web fetch are separate capabilities.

Run:

``` powershell
openclaw configure --section web
```

Follow the interactive prompts.

For the setup used in this walkthrough: 1. Enable `web_search`. 2.
Select **Ollama Web Search** if it is offered and you have completed the
required Ollama sign-in. 3. Enable `web_fetch` if you want the agent to
read a specific URL.

`web_fetch` reads a supplied URL; it is not a replacement for search.

After changing settings, check the Gateway:

``` powershell
openclaw gateway status
```

If needed:

``` powershell
openclaw gateway restart
```

------------------------------------------------------------------------

## 9. Create a Telegram Bot

### 9.1 Open BotFather

In Telegram, search for:

https://t.me/BotFather

Check that the account is the official `@BotFather`.

### 9.2 Create a bot

Send:

``` text
/newbot
```

BotFather asks for: 1. A display name, such as `My OpenClaw Assistant`.
2. A unique username ending in `bot`, such as `my_openclaw_demo_bot`.

BotFather returns a bot token.

It looks similar to:

``` text
123456789:AAExampleTokenValue
```

This is a secret credential. Do not share it in screenshots, GitHub,
source code, or public messages.

### 9.3 Add the bot to OpenClaw

In PowerShell, run the following command, replacing the placeholder with
your actual token:

``` powershell
openclaw channels add --channel telegram --token "YOUR_TELEGRAM_BOT_TOKEN"
```

Do not type the placeholder literally. Paste your own token between the
quotes.

The CLI stores the Telegram channel configuration. Treat your OpenClaw
configuration directory as sensitive because it can contain credentials.

### 9.4 Check Telegram channel status

``` powershell
openclaw channels status --probe
```

If the Gateway is already running, the configuration normally
hot-reloads. If the channel does not become ready, restart the Gateway:

``` powershell
openclaw gateway restart
```

Then check again:

``` powershell
openclaw channels status --probe
```

------------------------------------------------------------------------

## 10. Pair Your Telegram Account

Open Telegram and search for the bot username you created.

1.  Open the bot's chat.
2.  Press **Start** or send:

``` text
Hi
```

3.  Open PowerShell and list the pending pairing request:

``` powershell
openclaw pairing list telegram
```

4.  Copy the pairing code shown for your Telegram account.
5.  Approve that code:

``` powershell
openclaw pairing approve telegram YOUR_PAIRING_CODE
```

Replace `YOUR_PAIRING_CODE` with the actual code.

Pairing codes expire after one hour. If it expires, send another message
to the bot and repeat the pairing steps.

Now return to Telegram and send:

``` text
Hello! What can you do?
```

Your OpenClaw assistant should reply through the configured model.

**Security note:** Keep the default pairing policy for a personal bot.
Do not make the bot public with an unrestricted DM policy unless you
have deliberately designed and secured that setup.

------------------------------------------------------------------------

## 11. Optional: Add the Bot to a Telegram Group

For a first installation, test direct messages before adding group
access.

If you later add the bot to a group: - Keep group access restricted to
approved users/groups. - Configure the group ID under
`channels.telegram.groups`. - Telegram Privacy Mode may prevent the bot
from seeing ordinary group messages. Configure `/setprivacy` in
BotFather only if you understand the group visibility implications. -
Avoid opening the bot to every group member unless that is intentional.

Official access-control documentation:
https://docs.openclaw.ai/channels/telegram/access-control

------------------------------------------------------------------------

## 12. Daily Start and Stop Commands

### If you installed the Gateway background service

Start:

``` powershell
openclaw gateway start
```

Check:

``` powershell
openclaw gateway status
```

Open the UI:

``` powershell
openclaw dashboard
```

Stop:

``` powershell
openclaw gateway stop
```

### If you run the Gateway in the foreground

Start:

``` powershell
openclaw gateway
```

Keep that terminal open. Stop it with:

``` text
Ctrl + C
```

### Check Ollama

``` powershell
ollama list
ollama ps
```

Ollama must be available whenever OpenClaw needs a local model. Cloud
model use also needs internet access and valid Ollama cloud sign-in.

------------------------------------------------------------------------

## 13. Troubleshooting

### Problem: `ollama` is not recognized

-   Confirm Ollama is installed.
-   Close and reopen PowerShell.
-   Run:

``` powershell
ollama --version
```

### Problem: Ollama API is not reachable

Run:

``` powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

Open the Ollama desktop application and retry if the API is unavailable.

### Problem: OpenClaw cannot find the model

Run:

``` powershell
ollama list
openclaw models list --provider ollama
```

Select the exact model identifier that OpenClaw lists. A local tag such
as `gemma4:latest` is not interchangeable with a cloud tag such as
`gemma4:31b-cloud`.

### Problem: Telegram bot does not reply

Run:

``` powershell
openclaw channels status --probe
openclaw gateway status
openclaw pairing list telegram
```

Check that: - The bot token was entered correctly. - The Gateway is
running. - Your Telegram account has completed pairing. - Ollama or the
configured cloud provider is reachable.

If needed:

``` powershell
openclaw gateway restart
```

### Problem: Dashboard does not open

Run:

``` powershell
openclaw gateway status
```

Then try:

``` powershell
openclaw dashboard
```

You can also browse to:

http://127.0.0.1:18789/

### Problem: Gateway or plugin reports an error

Run:

``` powershell
openclaw doctor
openclaw gateway status
```

For diagnostics, follow the current official OpenClaw troubleshooting
documentation rather than deleting configuration files.

------------------------------------------------------------------------

## 14. Updating OpenClaw

Before updating, back up your OpenClaw configuration and workspace.

Then use the update command supported by your installation:

``` powershell
openclaw update
```

Check the version and health:

``` powershell
openclaw --version
openclaw doctor
openclaw gateway status
```

If the Gateway needs to be restarted:

``` powershell
openclaw gateway restart
```

------------------------------------------------------------------------

## 15. Important Security Practices

-   Never publish your Telegram bot token.
-   Never commit `.env` files, `openclaw.json`, credentials, or private
    pairing data to Git.
-   Keep Telegram DMs on pairing or a strict allowlist.
-   Do not expose port `18789` or Ollama port `11434` publicly without a
    secure access design.
-   Use a strong model for agents that can execute tools or access
    files.
-   Review tool permissions before enabling shell, browser, file, or
    other powerful tools.
-   Back up your OpenClaw configuration and workspace securely.
-   Treat messages received from Telegram as untrusted user input.

------------------------------------------------------------------------

## 16. Final Verification Checklist

Use this checklist after completing the setup:

-   [ ] Ollama is installed.
-   [ ] `ollama list` shows the expected model.
-   [ ] Ollama API responds at `http://localhost:11434/api/tags`.
-   [ ] OpenClaw is installed and `openclaw --version` works.
-   [ ] OpenClaw is configured to use the intended Ollama model.
-   [ ] `openclaw gateway status` reports a running Gateway.
-   [ ] `openclaw dashboard` opens the Control UI.
-   [ ] Telegram bot was created using `@BotFather`.
-   [ ] Telegram token was added to OpenClaw.
-   [ ] `openclaw channels status --probe` reports Telegram ready.
-   [ ] Telegram pairing was approved.
-   [ ] A message sent from Telegram receives an AI response.

------------------------------------------------------------------------

## 17. Official Documentation

-   OpenClaw documentation: https://docs.openclaw.ai/
-   OpenClaw installation: https://docs.openclaw.ai/install
-   OpenClaw getting started: https://docs.openclaw.ai/getting-started
-   OpenClaw Ollama setup:
    https://docs.openclaw.ai/providers/ollama/setup
-   OpenClaw Ollama configuration:
    https://docs.openclaw.ai/providers/ollama/configuration
-   OpenClaw Telegram setup:
    https://docs.openclaw.ai/channels/telegram/setup
-   OpenClaw Telegram access control:
    https://docs.openclaw.ai/channels/telegram/access-control
-   Ollama download: https://ollama.com/download
-   Ollama documentation: https://docs.ollama.com/

------------------------------------------------------------------------

## Quick Command Reference

``` powershell
# Check versions
ollama --version
openclaw --version
node --version

# Ollama
ollama list
ollama run gemma4
ollama ps
Invoke-RestMethod http://localhost:11434/api/tags

# OpenClaw model configuration
openclaw models list --provider ollama
openclaw models set ollama/gemma4:latest

# Gateway and dashboard
openclaw gateway install
openclaw gateway start
openclaw gateway status
openclaw dashboard

# Web tools
openclaw configure --section web

# Telegram
openclaw channels add --channel telegram --token "YOUR_TELEGRAM_BOT_TOKEN"
openclaw channels status --probe
openclaw pairing list telegram
openclaw pairing approve telegram YOUR_PAIRING_CODE

# Diagnostics
openclaw doctor
openclaw gateway restart
```

**End of guide.**
