

# Get started with Amazon Bedrock AgentCore
<a name="agentcore-get-started-cli"></a>

This quickstart gets you from zero to a running agent in a few minutes using the AgentCore CLI. You will install the CLI, scaffold a project, test locally, deploy to AWS, and invoke your agent.

Two ways to build an agent on AgentCore, same CLI:
+  **Code-based agent** (default, GA). You write the agent loop in Python using a framework you already know (Strands, LangGraph, Google ADK, or OpenAI Agents), and deploy it to AgentCore Runtime. Full control over orchestration logic.
+  **Managed harness** (preview). You declare the agent in a config file (model, prompt, tools, memory) and AgentCore runs the loop for you. No framework, no orchestration code. Good path when you want the fastest route from idea to a running agent. [Learn more](harness.md).

This page walks through the code-based flow. For harness, see [What is the AgentCore harness](harness.md).

## Prerequisites
<a name="agentcore-cli-prerequisites"></a>
+  **Node.js 20 or later.** The AgentCore CLI is distributed as an npm package. Check with `node --version`. Install from [nodejs.org](https://nodejs.org) if needed.
+  **npm.** Included with Node.js.
+  **An AWS account with credentials configured.** Configure via AWS CLI, environment variables, or an AWS profile. See [Configuring the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).
+  **Python 3.10 or later** (for agent code). Check with `python3 --version`.
+  **IAM permissions.** Your identity needs permissions to make AgentCore API calls and to assume the CDK bootstrap roles used during deployment. See [AgentCore CLI IAM Permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security-iam.html).

## Step 1: Install the AgentCore CLI
<a name="agentcore-cli-install"></a>

```
npm install -g @aws/agentcore
```

Verify:

```
agentcore --version
```

To update later, rerun the install command or `agentcore update`. Source and issues: [agentcore-cli on GitHub](https://github.com/aws/agentcore-cli).

### Opt into the preview channel
<a name="agentcore-cli-preview"></a>

To access preview capabilities (harness, config-based agents, in-progress features), install the preview channel:

```
npm install -g @aws/agentcore@preview
```

The preview channel is the same CLI with preview features enabled. Stable commands behave identically. See [What is the AgentCore harness](harness.md) for what the preview unlocks.

## Step 2: Create your project
<a name="agentcore-cli-create"></a>

```
agentcore create
```

The interactive wizard asks you for:
+  **Framework** - Strands Agents (recommended), LangChain/LangGraph, Google Agent Development Kit, or OpenAI Agents SDK
+  **Model provider** - Amazon Bedrock, Anthropic, OpenAI, or Gemini
+  **Memory** - None, short-term only, or long-term and short-term
+  **Build type** - CodeZip (default) or Container

You can also pass flags directly:

```
agentcore create \
  --name MyAgent \
  --framework Strands \
  --model-provider Bedrock \
  --memory none \
  --build CodeZip
```

**Note**  
 **Preview alternative.** With the preview CLI installed, you can scaffold a config-based [harness](harness.md) instead of a code-based agent. The CLI surfaces this as a choice in the wizard. Harness is the fastest path to a running agent because there is no framework or orchestration code to write.

### Project structure
<a name="agentcore-cli-project-structure"></a>

 `agentcore create` generates:

```
MyAgent/
├── agentcore/
│   ├── agentcore.json      # Project and resource configuration
│   ├── aws-targets.json    # Deployment target (account and region)
│   └── cdk/                # CDK infrastructure (auto-managed)
└── app/
    └── MyAgent/            # Your agent code
        ├── main.py         # Agent entrypoint
        ├── pyproject.toml  # Python dependencies
        └── ...
```

Key files:
+  `agentcore/agentcore.json` - the main config. Defines your agents, memory stores, gateways, credentials, and other resources. Managed by `agentcore add` and `agentcore remove`.
+  `app/` - your agent code. Each agent gets its own subdirectory with an entrypoint and a `pyproject.toml`.
+  `agentcore/aws-targets.json` - the AWS account and region for deployment.

## Step 3: Test locally
<a name="agentcore-cli-test"></a>

```
cd MyAgent
agentcore dev
```

 `agentcore dev` creates a Python virtual environment, installs dependencies, starts a local server with hot reload, and opens the **agent inspector** in your browser so you can chat with the agent, inspect traces, and browse project resources. Code changes are picked up automatically.

Useful flags:
+  `--no-browser` - use the terminal-based TUI instead of the browser inspector.
+  `--no-traces` - disable writing traces to `agentcore/.cli/traces`.
+  `--logs` - tail server logs in non-interactive mode.
+  `--port <N>` - pin the dev port (default 8080 for HTTP, 8000 for MCP, 9000 for A2A; auto-increments if busy).

## Step 4: Deploy your agent
<a name="agentcore-cli-deploy"></a>

```
agentcore deploy
```

Deploy:

1. Packages your code into a zip artifact (or builds a container if `--build Container`)

1. Uses AWS CDK under the hood to synthesize and provision resources

1. Creates an AgentCore Runtime endpoint for your agent

1. Configures CloudWatch logging and observability

First deploy takes a few minutes while CDK bootstraps your account. Subsequent deploys are faster.

Preview what will change without deploying:

```
agentcore deploy --plan
```

Check status:

```
agentcore status
```

## Step 5: Invoke your deployed agent
<a name="agentcore-cli-invoke"></a>

```
agentcore invoke --prompt "Hello, what can you do?"
```

That’s the loop. Iterate on `app/MyAgent/main.py`, test with `agentcore dev`, deploy with `agentcore deploy`, invoke with `agentcore invoke`.

## Add capabilities to your project
<a name="agentcore-cli-add-capabilities"></a>

 `agentcore add` manages resources in `agentcore.json`. Run it without arguments for the interactive menu, or target a resource directly.

```
agentcore add memory        # Store conversation context
agentcore add agent         # Add a second agent to the same project
agentcore add gateway       # Connect external APIs/tools through Gateway
agentcore add credential    # Add an API key for a non-Bedrock provider
agentcore add evaluator     # Quality evaluation
```

Each add command scaffolds the config and prompts for required values. After adding, run `agentcore deploy` to provision.

Deep dives for the capabilities you can attach:
+  [AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html) - short-term and long-term memory, retrieval strategies
+  [AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html) - governed connectivity to APIs and MCP servers
+  [AgentCore Browser](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-tool.html) - managed web browsing for agents
+  [AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html) - sandboxed code execution
+  [AgentCore Identity](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity.html) - OAuth, API key credential providers, workload identity
+  [AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html) - traces, logs, and metrics in CloudWatch
+  [AgentCore VPC](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html) - run agents inside your VPC

## View logs and traces
<a name="agentcore-cli-logs"></a>

```
# Stream recent logs
agentcore logs

# Filter
agentcore logs --since 30m --level error
agentcore logs --query "timeout"

# List recent traces
agentcore traces list

# Get a specific trace
agentcore traces get <trace-id>
```

## Clean up
<a name="agentcore-cli-cleanup"></a>

```
agentcore remove all
agentcore deploy
```

 `remove all` resets the configuration. The follow-up `deploy` detects the empty state and tears down the resources in your account.

## Next steps
<a name="agentcore-cli-next-steps"></a>
+  [What is the AgentCore harness](harness.md) - the config-based path to a running agent (preview). Use any model, connect to tools, persist state, deploy in your VPC, and graduate to code when you need it.
+  [AgentCore code samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples) - end-to-end examples across frameworks and capabilities.