# Contributing to agentguard

Thank you for your interest in contributing to agentguard. This document will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rigvedrs/agentguard.git
cd agentguard

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_guard.py

# Run with coverage
pytest --cov=agentguard --cov-report=term-missing
```

## Code Quality

```bash
# Type checking
mypy src/agentguard

# Linting
ruff check src/agentguard

# Auto-format
ruff format src/agentguard
```

## Project Structure

```
src/agentguard/
├── core/          # @guard decorator, types, trace store, registry
├── validators/    # Schema, hallucination, semantic, custom validators
├── guardrails/    # Circuit breaker, rate limiter, budget, retry, timeout
├── testing/       # Trace recorder, replayer, test generator, assertions
├── integrations/  # OpenAI, Anthropic, LangChain, MCP wrappers
├── reporting/     # Console and JSON reporting
└── cli/           # Command-line interface
```

## How to Contribute

### Reporting Bugs

Open an issue with:
- Python version
- agentguard version
- Minimal reproduction steps
- Expected vs actual behaviour

### Suggesting Features

Open an issue with `[Feature]` in the title. Describe:
- The problem you're solving
- Your proposed solution
- Any alternatives you've considered

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add or update tests
5. Run `pytest` and `mypy` to ensure everything passes
6. Commit with a descriptive message
7. Push and open a PR

### Code Style

- Type hints on all public functions and methods
- Docstrings on all public classes and functions
- Follow existing naming conventions
- Minimal dependencies — propose new dependencies in the issue before adding

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
