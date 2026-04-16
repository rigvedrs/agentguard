# Contributing to agentguard

Thank you for considering a contribution to agentguard. This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/rigvedrs/agentguard.git
cd agentguard

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"
```

Verify the setup:

```bash
pytest
```

All tests should pass before you start making changes.

---

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_guard.py

# Run with coverage
pytest --cov=agentguard --cov-report=html

# Run only tests matching a pattern
pytest -k "test_circuit"

# Run with verbose output
pytest -v
```

---

## Code Style

agentguard uses:
- **ruff** for linting and formatting
- **mypy** for static type checking

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/agentguard/
```

All three must pass before submitting a PR. The CI pipeline checks them automatically.

---

## Project Structure

```
agentguard/
├── src/agentguard/
│   ├── __init__.py               # Public API exports
│   ├── core/
│   │   ├── guard.py              # @guard decorator, GuardedTool
│   │   ├── types.py              # ToolCall, ToolResult, GuardConfig, etc.
│   │   ├── trace.py              # TraceStore, TraceRecorder
│   │   └── registry.py          # ToolRegistry
│   ├── guardrails/
│   │   ├── circuit_breaker.py   # Circuit breaker state machine
│   │   ├── rate_limiter.py      # Token bucket rate limiter
│   │   ├── budget.py            # Budget enforcement
│   │   ├── retry.py             # Retry with exponential backoff
│   │   └── timeout.py           # Thread/asyncio timeout
│   ├── validators/
│   │   ├── schema.py            # Type-hint + Pydantic validation
│   │   ├── hallucination.py     # Multi-signal hallucination detector
│   │   ├── semantic.py          # Per-tool semantic validation
│   │   └── custom.py            # Custom validator pipeline
│   ├── integrations/
│   │   ├── openai_integration.py
│   │   ├── anthropic_integration.py
│   │   ├── langchain_integration.py
│   │   ├── mcp_integration.py
│   │   ├── openai_compatible.py
│   │   ├── crewai_integration.py
│   │   └── autogen_integration.py
│   ├── testing/
│   │   ├── assertions.py        # assert_tool_call()
│   │   ├── generator.py         # TestGenerator
│   │   ├── recorder.py          # TraceRecorder
│   │   └── replayer.py          # TraceReplayer
│   ├── reporting/
│   │   ├── console.py           # Rich terminal output
│   │   └── json_reporter.py     # JSON report generator
│   ├── cli/                     # Command-line interface
│   └── middleware.py            # MiddlewarePipeline
├── tests/
│   ├── test_guard.py
│   ├── test_guardrails.py
│   ├── test_validators.py
│   ├── test_integrations.py
│   ├── test_testing.py
│   ├── test_crewai.py
│   └── test_autogen.py
└── docs/                        # Documentation (MkDocs)
```

---

## Adding a New Integration

1. Create `src/agentguard/integrations/myframework_integration.py`
2. Use `try/except ImportError` for the optional dependency
3. Follow the pattern of `langchain_integration.py` or `crewai_integration.py`
4. Export from `integrations/__init__.py`
5. Write tests in `tests/test_myframework.py` — mock the framework dependency
6. Write documentation in `docs/integrations/myframework.md`

**Template:**

```python
"""MyFramework integration for agentguard.

MyFramework is an optional dependency. Import error is caught gracefully.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.types import GuardConfig

try:
    import myframework  # type: ignore[import]
    _MYFRAMEWORK_AVAILABLE = True
except ImportError:
    myframework = None
    _MYFRAMEWORK_AVAILABLE = False


class GuardedMyFrameworkTool:
    """agentguard-protected MyFramework tool."""
    
    def __init__(self, tool: Any, config: Optional[GuardConfig] = None) -> None:
        self.name = getattr(tool, "name", None) or getattr(tool, "__name__", "unnamed")
        self.description = getattr(tool, "description", "") or (tool.__doc__ or "").splitlines()[0]
        cfg = config or GuardConfig()
        fn = self._extract_callable(tool)
        self.guarded_fn = guard(fn, config=cfg) if not isinstance(fn, GuardedTool) else fn
    
    def _extract_callable(self, tool: Any) -> Callable:
        if callable(tool):
            return tool
        raise TypeError(f"Cannot extract callable from {tool!r}")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.guarded_fn(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"GuardedMyFrameworkTool(name={self.name!r})"


def guard_myframework_tools(
    tools: List[Any],
    config: Optional[GuardConfig] = None,
) -> List[GuardedMyFrameworkTool]:
    return [GuardedMyFrameworkTool(t, config=config) for t in tools]
```

---

## Adding a New Guardrail

1. Create `src/agentguard/guardrails/myguardrail.py`
2. Add configuration fields to `GuardConfig` in `core/types.py`
3. Wire it up in `core/guard.py`'s execution pipeline
4. Write tests in `tests/test_guardrails.py`
5. Document in `docs/guides/`

---

## Submitting a Pull Request

1. **Fork** the repository on GitHub
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write code, tests, and documentation
4. Run `pytest`, `ruff check`, `mypy` — all must pass
5. Commit with a clear message: `git commit -m "feat: add X integration"`
6. Push and open a PR against `main`

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated if applicable
- [ ] `CHANGELOG.md` updated with a brief description
- [ ] New optional dependencies added to `pyproject.toml` under `[project.optional-dependencies]`

---

## Reporting Bugs

Open a [GitHub issue](https://github.com/rigvedrs/agentguard/issues) with:

- agentguard version (`pip show agentguard`)
- Python version (`python --version`)
- Minimal reproducible example
- Full traceback

---

## Questions?

Open a [GitHub Discussion](https://github.com/rigvedrs/agentguard/discussions) for questions that aren't bug reports.
