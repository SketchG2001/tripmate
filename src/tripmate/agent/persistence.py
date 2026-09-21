from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("tripmate.agent.models", "TraceEvent"),
            ("tripmate.agent.models", "ToolCall"),
        ],
        pickle_fallback=False,
    )
