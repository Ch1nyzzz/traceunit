"""No-model fixture used to smoke-test the isolated AppWorld worker."""


def solve(world):
    output = world.execute("print('traceunit appworld smoke')")
    return {
        "steps": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
        "transcript": [
            {"role": "assistant", "content": "print('traceunit appworld smoke')"},
            {"role": "user", "content": str(output)},
        ],
    }
