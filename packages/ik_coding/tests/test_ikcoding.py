"""Tests for ik_coding — real, no mocks."""

from __future__ import annotations

import pytest

from ik_coding import (
    CodeMode,
    CodeResult,
    CodeTask,
    CodingError,
    Language,
    TestRun,
    execute_coding_task,
    extract_code_blocks,
    looks_like_python,
    plan_code_strategy,
    run_code_safely,
    score_iteration,
    validate_code_request,
)


class TestValidation:
    def test_valid_task(self):
        task = CodeTask(language="python", instruction="add two numbers")
        validate_code_request(task)  # should not raise

    def test_empty_instruction(self):
        task = CodeTask(language="python", instruction="")
        with pytest.raises(CodingError):
            validate_code_request(task)

    def test_whitespace_instruction(self):
        task = CodeTask(language="python", instruction="   ")
        with pytest.raises(CodingError):
            validate_code_request(task)

    def test_unsupported_language(self):
        task = CodeTask(language="cobol", instruction="hello")
        with pytest.raises(CodingError, match="unsupported language"):
            validate_code_request(task)

    def test_unsupported_mode(self):
        task = CodeTask(language="python", instruction="hello", mode="nonsense")
        with pytest.raises(CodingError, match="unsupported mode"):
            validate_code_request(task)

    def test_timeout_out_of_range(self):
        task = CodeTask(language="python", instruction="x", timeout_s=0)
        with pytest.raises(CodingError, match="timeout_s"):
            validate_code_request(task)
        task2 = CodeTask(language="python", instruction="x", timeout_s=1000)
        with pytest.raises(CodingError, match="timeout_s"):
            validate_code_request(task2)

    def test_max_iterations_out_of_range(self):
        task = CodeTask(language="python", instruction="x", max_iterations=0)
        with pytest.raises(CodingError, match="max_iterations"):
            validate_code_request(task)


class TestExtractCodeBlocks:
    def test_single_block(self):
        text = "Here is the code:\n```python\nprint('hi')\n```\nDone."
        blocks = extract_code_blocks(text, "python")
        assert blocks == ["print('hi')\n"]

    def test_multiple_blocks(self):
        text = "```python\nx = 1\n```\nand\n```python\ny = 2\n```"
        blocks = extract_code_blocks(text, "python")
        assert len(blocks) == 2
        assert "x = 1" in blocks[0]
        assert "y = 2" in blocks[1]

    def test_no_language_filter(self):
        text = "```python\na\n```\n```js\nb\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2

    def test_no_blocks(self):
        assert extract_code_blocks("no code here") == []

    def test_wrong_language_filter(self):
        text = "```python\nx = 1\n```"
        assert extract_code_blocks(text, "rust") == []


class TestLooksLikePython:
    def test_def_keyword(self):
        assert looks_like_python("def foo(): pass")

    def test_import_keyword(self):
        assert looks_like_python("import os\nos.getcwd()")

    def test_indented(self):
        assert looks_like_python("    x = 1\n    return x")

    def test_empty(self):
        assert not looks_like_python("")

    def test_whitespace(self):
        assert not looks_like_python("   ")


class TestPlanStrategy:
    def test_single_shot(self):
        task = CodeTask(language="python", instruction="x", mode=CodeMode.SINGLE_SHOT.value)
        steps = plan_code_strategy(task)
        assert any("generate code" in s for s in steps)
        assert any("validate syntax" in s for s in steps)

    def test_test_driven(self):
        task = CodeTask(language="python", instruction="x", mode=CodeMode.TEST_DRIVEN.value)
        steps = plan_code_strategy(task)
        assert any("test cases" in s for s in steps)

    def test_iterative(self):
        task = CodeTask(
            language="python", instruction="x", mode=CodeMode.ITERATIVE.value, max_iterations=5
        )
        steps = plan_code_strategy(task)
        assert any("max=5" in s for s in steps)

    def test_with_context(self):
        task = CodeTask(language="python", instruction="x", context="use class A")
        steps = plan_code_strategy(task)
        assert any("context" in s for s in steps)


class TestScoreIteration:
    def test_all_pass_short_code(self):
        # "def f(): pass" → 1 line, quality=0.05, pass_rate=1.0 → 0.81
        results = [TestRun("t1", True), TestRun("t2", True)]
        assert score_iteration("def f(): pass", results) == round(0.8 + 0.2 * 0.05, 4)

    def test_all_fail(self):
        # pass_rate=0.0, quality=0.05 → 0.01
        results = [TestRun("t1", False), TestRun("t2", False)]
        assert score_iteration("def f(): pass", results) == 0.01

    def test_half_pass(self):
        # pass_rate=0.5, code has 2 lines, quality=0.1 → 0.42
        results = [TestRun("t1", True), TestRun("t2", False)]
        score = score_iteration("def f(): pass\nclass A: pass", results)
        assert score == 0.42

    def test_no_tests_with_code(self):
        assert score_iteration("def f(): pass", []) == 0.5

    def test_no_tests_no_code(self):
        assert score_iteration("", []) == 0.0


class TestRunCodeSafely:
    def test_python_success(self):
        result = run_code_safely("print(1+1)", "python", timeout_s=5)
        assert result.passed
        assert "2" in result.message

    def test_python_failure(self):
        result = run_code_safely("raise RuntimeError('boom')", "python", timeout_s=5)
        assert not result.passed
        assert "boom" in result.message

    def test_python_timeout(self):
        result = run_code_safely("import time; time.sleep(10)", "python", timeout_s=1)
        assert not result.passed
        assert "timeout" in result.message

    def test_unsupported_language_local(self):
        result = run_code_safely("echo hi", "bash", timeout_s=5)
        assert not result.passed
        assert "ik_sandbox" in result.message


class TestExecuteTask:
    def test_successful_python(self):
        task = CodeTask(language="python", instruction="compute factorial")
        result = execute_coding_task(
            task, "def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)\nprint(fact(5))"
        )
        assert result.success
        assert "120" in result.output
        assert result.iterations == 1

    def test_failing_python(self):
        task = CodeTask(language="python", instruction="x")
        result = execute_coding_task(task, "raise ValueError('nope')")
        assert not result.success
        assert "nope" in result.error

    def test_empty_code(self):
        task = CodeTask(language="python", instruction="x")
        result = execute_coding_task(task, "")
        assert not result.success
        assert "empty" in result.error

    def test_test_driven_runs_test_code(self):
        task = CodeTask(
            language="python",
            instruction="x",
            mode=CodeMode.TEST_DRIVEN.value,
            test_code="print('tests pass')",
        )
        result = execute_coding_task(task, "print('impl')", "print('tests pass')")
        assert result.success
        assert len(result.test_results) == 2

    def test_iterative_with_failing_test(self):
        task = CodeTask(
            language="python",
            instruction="x",
            mode=CodeMode.ITERATIVE.value,
        )
        result = execute_coding_task(task, "raise Exception('fail')")
        assert not result.success
        assert result.iterations == 1
