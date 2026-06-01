"""
Reward function for LiveCodeBench: execution-based evaluation.

Adapted from SDPO's code execution reward (verl/utils/reward_score/feedback/code.py).
Executes the generated code against test cases in a sandboxed subprocess with
timeout protection.

The ground_truth should be a JSON string with format:
{
    "inputs": [...],
    "outputs": [...],
    "testtype": "functional" | "stdin",
    "fn_name": "function_name",
    "time_limit": 6
}

Usage as verl reward function:
    reward.custom_reward_function.path = src/rewards/livecodebench_reward.py
    reward.custom_reward_function.name = compute_score
"""

import io
import json
import multiprocessing
import re
import sys
import time
import traceback
from typing import Optional


DEFAULT_TIMEOUT = 6
MAX_TEST_CASES = 20


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from response (markdown blocks or raw)."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return max(blocks, key=len)

    # Fallback: look for def/class statements
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("def ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    return None


def _run_functional_test(completion: str, test_input, test_output, fn_name: str) -> tuple[bool, str]:
    """Run a functional (call-based) test case."""
    namespace = {"__builtins__": __builtins__}
    try:
        exec(compile(completion, "<solution>", "exec"), namespace)
    except Exception as e:
        return False, f"CompileError: {e}"

    # Find the function
    func = namespace.get(fn_name)
    if func is None:
        # Try to find any defined function
        import ast
        try:
            tree = ast.parse(completion)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func = namespace.get(node.name)
                    if func is not None:
                        break
        except Exception:
            pass

    if func is None or not callable(func):
        return False, "NoFunctionFound"

    try:
        if isinstance(test_input, dict):
            result = func(**test_input)
        elif isinstance(test_input, list):
            result = func(*test_input)
        else:
            args = [json.loads(x) for x in str(test_input).split()]
            result = func(*args)

        expected = json.loads(test_output) if isinstance(test_output, str) else test_output
        if json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True):
            return True, str(result)
        return False, str(result)
    except Exception as e:
        return False, f"RuntimeError: {e}"


def _run_stdin_test(completion: str, test_input: str, test_output: str) -> tuple[bool, str]:
    """Run a stdin/stdout test case."""
    old_stdout, old_stdin = sys.stdout, sys.stdin
    output_buffer = io.StringIO()
    try:
        sys.stdout = output_buffer
        sys.stdin = io.StringIO(test_input)
        namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
        exec(compile(completion, "<solution>", "exec"), namespace)
        actual = output_buffer.getvalue().strip()
        expected = test_output.strip()
        return actual == expected, actual
    except Exception as e:
        return False, f"RuntimeError: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


def _run_single_test(test_cases: dict, completion: str, test_idx: int, send_conn):
    """Run a single test in a subprocess."""
    test_type = test_cases.get("testtype", "functional")
    fn_name = test_cases.get("fn_name", "")
    test_input = test_cases["inputs"][test_idx]
    test_output = test_cases["outputs"][test_idx]

    try:
        if test_type == "functional":
            passed, output = _run_functional_test(completion, test_input, test_output, fn_name)
        elif test_type == "stdin":
            passed, output = _run_stdin_test(completion, test_input, test_output)
        else:
            passed, output = False, f"UnknownTestType: {test_type}"
    except Exception as e:
        passed, output = False, f"Error: {e}"

    send_conn.send({"test_idx": test_idx, "passed": passed, "output": output})
    send_conn.close()


def run_tests(test_cases: dict, completion: str, timeout: float = DEFAULT_TIMEOUT, max_tests: int = MAX_TEST_CASES) -> list[dict]:
    """Run test cases with subprocess isolation and timeout."""
    n_tests = min(max_tests, len(test_cases["inputs"]))
    records = []

    for test_idx in range(n_tests):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=_run_single_test, args=(test_cases, completion, test_idx, child_conn))
        p.start()
        child_conn.close()

        if parent_conn.poll(timeout):
            try:
                result = parent_conn.recv()
            except Exception:
                result = {"test_idx": test_idx, "passed": False, "output": "PipeError"}
        else:
            result = {"test_idx": test_idx, "passed": False, "output": "Timeout"}

        records.append(result)
        p.join(timeout=0)
        if p.is_alive():
            p.kill()
            p.join()

    return records


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute reward for LiveCodeBench via code execution.

    Returns dict with score (fraction of test cases passed), feedback, etc.
    """
    if not solution_str or not ground_truth:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Empty input."}

    completion = extract_code(solution_str)
    if completion is None:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "No code block found in response."}

    try:
        test_cases = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Failed to parse test cases."}

    if "inputs" not in test_cases or "outputs" not in test_cases:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "Invalid test cases format."}

    timeout = float(test_cases.get("time_limit", DEFAULT_TIMEOUT))
    records = run_tests(test_cases, completion, timeout=timeout)

    if not records:
        return {"score": 0.0, "acc": 0.0, "pred": "", "feedback": "No test results."}

    n_passed = sum(1 for r in records if r["passed"])
    n_total = len(records)
    score = n_passed / n_total

    # Build feedback from failed tests
    failed = [r for r in records if not r["passed"]]
    feedback_parts = []
    for r in failed[:2]:  # Show at most 2 failures
        output = str(r.get("output", ""))[:200]
        feedback_parts.append(f"Test {r['test_idx']}: {output}")
    feedback = "; ".join(feedback_parts) if feedback_parts else ""

    return {
        "score": score,
        "acc": score,
        "pred": f"{n_passed}/{n_total} passed",
        "feedback": feedback,
    }
