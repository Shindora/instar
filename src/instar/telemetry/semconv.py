"""Every OpenTelemetry name instar reads. The GenAI conventions are still in development, so
nothing outside this module spells one out."""

OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
SYSTEM = "gen_ai.system"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
TOOL_NAME = "gen_ai.tool.name"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
AGENT_NAME = "gen_ai.agent.name"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
ERROR_TYPE = "error.type"

LLM_OPERATIONS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
TOOL_OPERATIONS = frozenset({"execute_tool"})
AGENT_OPERATIONS = frozenset({"invoke_agent", "create_agent"})

KEPT_ATTRIBUTES = frozenset(
    {
        OPERATION,
        PROVIDER,
        SYSTEM,
        REQUEST_MODEL,
        RESPONSE_MODEL,
        TOOL_NAME,
        AGENT_NAME,
        INPUT_TOKENS,
        OUTPUT_TOKENS,
        CACHE_READ_TOKENS,
        ERROR_TYPE,
    }
)

SESSION = "instar.session"
CALL_INDEX = "instar.call_index"
FAULT = "instar.fault"
COST = "instar.cost_usd"
TOOL_ARGUMENTS_HASH = "instar.tool_arguments_sha256"

TRACES_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
TRACES_PROTOCOL = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
RESOURCE_ATTRIBUTES = "OTEL_RESOURCE_ATTRIBUTES"
