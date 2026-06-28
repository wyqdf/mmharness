import subprocess
import shutil
from pathlib import Path

MODEL = "claude-opus-4-6"

claude_path = (
    shutil.which("claude.cmd")
    or shutil.which("claude.exe")
    or shutil.which("claude")
)

print("claude_path:", claude_path)

if claude_path is None:
    raise RuntimeError("Python 找不到 claude，请检查 PATH，或手动填 claude.cmd 完整路径")

long_prompt = "只回复 OK。\n\n" + ("这是很长的填充文本。\n" * 5000)

print("prompt chars:", len(long_prompt))


def run_argv():
    print("\n=== argv 方式 ===")
    cmd = [
        claude_path,
        "-p",
        long_prompt,
        "--model",
        MODEL,
    ]

    try:
        r = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=180,
        )
        print("returncode:", r.returncode)
        print("stdout:", repr(r.stdout[:500]))
        print("stderr:", repr(r.stderr[:1000]))
    except Exception as e:
        print("EXCEPTION:", repr(e))


def run_stdin():
    print("\n=== stdin 方式 ===")
    cmd = [
        claude_path,
        "-p",
        "--model",
        MODEL,
    ]

    try:
        r = subprocess.run(
            cmd,
            input=long_prompt,
            text=True,
            capture_output=True,
            timeout=180,
        )
        print("returncode:", r.returncode)
        print("stdout:", repr(r.stdout[:500]))
        print("stderr:", repr(r.stderr[:1000]))
    except Exception as e:
        print("EXCEPTION:", repr(e))


run_argv()
run_stdin()