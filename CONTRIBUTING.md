# Contributing to BulkRNA Agent

Thank you for your interest in contributing to BulkRNA Agent! 🧬

## How to Contribute

### Reporting Issues

Found a bug or have a feature request?

1. Check if the issue already exists
2. Open a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System info (OS, Python version)
   - Log excerpts if relevant

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/bulkrna-agent.git
   cd bulkrna-agent
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation

5. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   flake8 bulkrna_agent/
   
   # Run example
   python examples/example_usage.py
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue with..."
   ```

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then create a PR on GitHub with:
   - Clear description of changes
   - Related issue numbers
   - Screenshots (if UI changes)

## Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions focused and small
- Use meaningful variable names

**Example:**
```python
def calculate_fold_change(
    counts: pd.DataFrame,
    condition_a: str,
    condition_b: str
) -> pd.DataFrame:
    """
    Calculate fold change between conditions.
    
    Args:
        counts: Count matrix with samples as columns
        condition_a: Name of first condition
        condition_b: Name of second condition
    
    Returns:
        DataFrame with fold changes
    """
    # Implementation
    pass
```

### Testing

Add tests for new features:

```python
def test_new_feature():
    """Test the new feature"""
    # Setup
    config = BulkRNAConfig()
    tool = NewTool(config)
    
    # Execute
    result = tool.execute(param="value")
    
    # Assert
    assert result["status"] == "success"
    assert "expected_key" in result
```

Run tests:
```bash
pytest tests/ -v
pytest tests/test_specific.py::test_function -v
```

### Logging

Use proper logging:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting operation")
    try:
        # Code
        logger.debug("Debug info")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
```

### Documentation

Update documentation for:
- New features
- API changes
- Configuration options
- Breaking changes

**Files to update:**
- `README.md` - Main documentation
- `docs/API.md` - API reference
- `docs/TUTORIAL.md` - Tutorial updates
- Docstrings in code

## Areas for Contribution

### High Priority

- [ ] Additional visualization tools
- [ ] More comprehensive tests
- [ ] Performance optimization
- [ ] Better error messages
- [ ] Docker container

### Features

- [ ] Support for single-cell RNA-seq
- [ ] Time-series analysis
- [ ] GSEA implementation
- [ ] Interactive plots
- [ ] PDF report generation
- [ ] Alternative normalization methods
- [ ] Support for more organisms

### Improvements

- [ ] Faster enrichment methods
- [ ] Caching for LLM responses
- [ ] Better prompt engineering
- [ ] More analysis tools
- [ ] Batch processing mode

### Documentation

- [ ] Video tutorials
- [ ] More examples
- [ ] Jupyter notebook examples
- [ ] API documentation improvements
- [ ] Troubleshooting guide expansion

## Project Structure

```
bulkrna_agent/
├── __init__.py          # Package initialization
├── config.py            # Configuration classes
├── llm.py              # LLM interfaces
├── agent.py            # ReAct agent
├── tools.py            # Analysis tools
├── mcp_server.py       # MCP server integration
└── web_interface.py    # Gradio interface

docs/                   # Documentation
examples/               # Example scripts
tests/                  # Test suite
```

## Commit Message Convention

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Build/tooling changes

**Examples:**
```
feat: add support for edgeR analysis
fix: resolve memory leak in large datasets
docs: update tutorial with new features
test: add tests for enrichment tool
```

## Code Review Process

1. All PRs require review
2. Tests must pass
3. Documentation must be updated
4. Code style must be consistent
5. No merge conflicts

## Questions?

- Open a GitHub issue for questions
- Join discussions
- Check existing documentation

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making BulkRNA Agent better! 🚀
