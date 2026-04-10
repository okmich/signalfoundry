#!/usr/bin/env python3
"""
Build script for signalfoundry mono-repo.

Handles cleanup, testing, linting, formatting, type checking, and building
of all or specific sub-projects.

Usage:
    python build.py --<action> [project]

Actions:
    --clean      - Remove build artifacts and cache files
    --test       - Run pytest on projects
    --build      - Build distribution packages
    --lint       - Run ruff linter
    --format     - Format code with black
    --typecheck  - Run mypy type checker
    --sync       - Sync dependencies with uv
    --all        - Run full pipeline (clean, sync, format, lint, typecheck, test, build)

Examples:
    python build.py --clean
    python build.py --test core
    python build.py --build
    python build.py --all features
    python build.py --test core --no-coverage
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Script configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECTS_DIR = SCRIPT_DIR

# All sub-projects in dependency order
ALL_PROJECTS = [
    "core",
    "utils",
    "mt5",
    "features",
    "labelling",
    "ml",
    "neural-net",
    "pipeline",
    "research",
]

# Patterns to clean
CLEAN_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "dist",
    "build",
    ".coverage",
    "htmlcov",
]

CLEAN_FILE_PATTERNS = ["*.pyc", "*.pyo"]


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(message: str) -> None:
    """Print a section header."""
    print()
    print(f"{Colors.CYAN}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD} {message}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 50}{Colors.RESET}")


def print_subheader(message: str) -> None:
    """Print a sub-section header."""
    print()
    print(f"{Colors.YELLOW}--- {message} ---{Colors.RESET}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[OK] {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[ERROR] {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.GRAY}  {message}{Colors.RESET}")


def run_command(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        raise e


def get_project_path(project: str) -> Path:
    """Get the path to a project directory."""
    return PROJECTS_DIR / project


def action_clean(projects: list[str]) -> bool:
    """Clean build artifacts and cache files."""
    print_header("Cleaning Projects")

    for proj in projects:
        print_subheader(f"Cleaning {proj}")
        proj_path = get_project_path(proj)

        if not proj_path.exists():
            print_error(f"Project path not found: {proj_path}")
            continue

        # Clean directory patterns
        for pattern in CLEAN_PATTERNS:
            for item in proj_path.rglob(pattern):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print_info(f"Removed: {item}")
                except Exception:
                    pass  # Ignore locked files

        # Clean file patterns
        for pattern in CLEAN_FILE_PATTERNS:
            for item in proj_path.rglob(pattern):
                try:
                    item.unlink()
                    print_info(f"Removed: {item}")
                except Exception:
                    pass

        print_success(f"Cleaned {proj}")

    # Clean utilities-level artifacts if cleaning all projects
    if set(projects) == set(ALL_PROJECTS):
        print_subheader("Cleaning utilities-level artifacts")
        for pattern in CLEAN_PATTERNS:
            item_path = SCRIPT_DIR / pattern
            if item_path.exists():
                try:
                    if item_path.is_dir():
                        shutil.rmtree(item_path)
                    else:
                        item_path.unlink()
                    print_info(f"Removed: {item_path}")
                except Exception:
                    pass
        print_success("Cleaned utilities artifacts")

    return True


def action_sync() -> bool:
    """Sync dependencies with uv."""
    print_header("Syncing Dependencies")

    try:
        run_command(["uv", "sync"], cwd=SCRIPT_DIR)
        print_success("Dependencies synced")
        return True
    except subprocess.CalledProcessError:
        print_error("Sync failed")
        return False


def action_test(projects: list[str], no_coverage: bool = False) -> bool:
    """Run tests for projects."""
    print_header("Running Tests")

    failed = []

    for proj in projects:
        print_subheader(f"Testing {proj}")
        proj_path = get_project_path(proj)
        test_path = proj_path / "tests"

        if not test_path.exists():
            print_info("No tests directory found, skipping...")
            continue

        # Check if there are any test files
        test_files = list(test_path.glob("test_*.py")) + list(test_path.glob("*_test.py"))
        test_files += list(test_path.rglob("test_*.py")) + list(test_path.rglob("*_test.py"))
        if not test_files:
            print_info("No test files found, skipping...")
            continue

        # Use python -m pytest to avoid script path issues on Windows
        relative_test_path = f"{proj}/tests"
        cmd = ["uv", "run", "python", "-m", "pytest", relative_test_path, "-v"]
        if not no_coverage:
            relative_src_path = f"{proj}/src"
            cmd.extend([f"--cov={relative_src_path}", "--cov-report=term-missing"])

        try:
            run_command(cmd, cwd=SCRIPT_DIR)
            print_success(f"Tests passed for {proj}")
        except subprocess.CalledProcessError:
            failed.append(proj)
            print_error(f"Tests failed for {proj}")

    if failed:
        print()
        print_error(f"Tests failed for: {', '.join(failed)}")
        return False

    return True


def action_build(projects: list[str]) -> bool:
    """Build distribution packages for projects."""
    print_header("Building Projects")

    failed = []

    for proj in projects:
        print_subheader(f"Building {proj}")
        proj_path = get_project_path(proj)

        if not proj_path.exists():
            print_error(f"Project path not found: {proj_path}")
            failed.append(proj)
            continue

        try:
            # Build from utilities root, specifying the project directory
            relative_proj_path = proj
            run_command(["uv", "build", "--directory", relative_proj_path], cwd=SCRIPT_DIR)
            print_success(f"Built {proj}")
        except subprocess.CalledProcessError:
            failed.append(proj)
            print_error(f"Build failed for {proj}")

    if failed:
        print()
        print_error(f"Build failed for: {', '.join(failed)}")
        return False

    return True


def action_lint(projects: list[str]) -> bool:
    """Run ruff linter on projects."""
    print_header("Linting Projects")

    failed = []

    for proj in projects:
        print_subheader(f"Linting {proj}")
        proj_path = get_project_path(proj)
        src_path = proj_path / "src"

        if not src_path.exists():
            print_info("No src directory found, skipping...")
            continue

        try:
            relative_src_path = f"{proj}/src"
            run_command(["uv", "run", "python", "-m", "ruff", "check", relative_src_path], cwd=SCRIPT_DIR)
            print_success(f"Lint passed for {proj}")
        except subprocess.CalledProcessError:
            failed.append(proj)
            print_error(f"Lint errors in {proj}")

    if failed:
        print()
        print_error(f"Lint failed for: {', '.join(failed)}")
        return False

    return True


def action_format(projects: list[str]) -> bool:
    """Format code with black."""
    print_header("Formatting Projects")

    for proj in projects:
        print_subheader(f"Formatting {proj}")
        proj_path = get_project_path(proj)

        paths_to_format = []
        if (proj_path / "src").exists():
            paths_to_format.append(f"{proj}/src")
        if (proj_path / "tests").exists():
            paths_to_format.append(f"{proj}/tests")

        if not paths_to_format:
            print_info("No src/tests directories found, skipping...")
            continue

        try:
            run_command(["uv", "run", "python", "-m", "black"] + paths_to_format, cwd=SCRIPT_DIR)
            print_success(f"Formatted {proj}")
        except subprocess.CalledProcessError:
            print_error(f"Format failed for {proj}")

    return True


def action_typecheck(projects: list[str]) -> bool:
    """Run mypy type checker on projects."""
    print_header("Type Checking Projects")

    failed = []

    for proj in projects:
        print_subheader(f"Type checking {proj}")
        proj_path = get_project_path(proj)
        src_path = proj_path / "src"

        if not src_path.exists():
            print_info("No src directory found, skipping...")
            continue

        try:
            relative_src_path = f"{proj}/src"
            run_command(
                ["uv", "run", "python", "-m", "mypy", relative_src_path, "--ignore-missing-imports"],
                cwd=SCRIPT_DIR,
            )
            print_success(f"Type check passed for {proj}")
        except subprocess.CalledProcessError:
            failed.append(proj)
            print_error(f"Type errors in {proj}")

    if failed:
        print()
        print_error(f"Type check failed for: {', '.join(failed)}")
        return False

    return True


def action_all(projects: list[str], no_coverage: bool = False) -> bool:
    """Run full build pipeline."""
    print_header("Running Full Build Pipeline")
    print(f"{Colors.MAGENTA}Target projects: {', '.join(projects)}{Colors.RESET}")

    steps = [
        ("clean", lambda: action_clean(projects)),
        ("sync", action_sync),
        ("format", lambda: action_format(projects)),
        ("lint", lambda: action_lint(projects)),
        ("typecheck", lambda: action_typecheck(projects)),
        ("test", lambda: action_test(projects, no_coverage)),
        ("build", lambda: action_build(projects)),
    ]

    for step_name, step_func in steps:
        if not step_func():
            print()
            print_error(f"Pipeline failed at step: {step_name}")
            return False

    print_header("Build Pipeline Complete")
    print_success("All steps completed successfully!")
    return True


def parse_project(value: str | None) -> list[str]:
    """Parse project argument, returning list of projects to target."""
    if value is None:
        return ALL_PROJECTS
    if value not in ALL_PROJECTS:
        print_error(f"Unknown project: {value}")
        print_info(f"Available projects: {', '.join(ALL_PROJECTS)}")
        sys.exit(1)
    return [value]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build script for signalfoundry mono-repo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Action flags - each takes an optional project name
    actions_group = parser.add_mutually_exclusive_group()

    actions_group.add_argument(
        "--clean",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Remove build artifacts and cache files",
    )

    actions_group.add_argument(
        "--test",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Run pytest on projects",
    )

    actions_group.add_argument(
        "--build",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Build distribution packages",
    )

    actions_group.add_argument(
        "--lint",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Run ruff linter",
    )

    actions_group.add_argument(
        "--format",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Format code with black",
    )

    actions_group.add_argument(
        "--typecheck",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Run mypy type checker",
    )

    actions_group.add_argument(
        "--sync",
        action="store_true",
        help="Sync dependencies with uv",
    )

    actions_group.add_argument(
        "--all",
        nargs="?",
        const="__all__",
        metavar="PROJECT",
        help="Run full pipeline (clean, sync, format, lint, typecheck, test, build)",
    )

    # Additional options
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting during tests",
    )

    args = parser.parse_args()

    # Determine which action was specified
    action = None
    project_arg = None

    if args.clean is not None:
        action = "clean"
        project_arg = args.clean
    elif args.test is not None:
        action = "test"
        project_arg = args.test
    elif args.build is not None:
        action = "build"
        project_arg = args.build
    elif args.lint is not None:
        action = "lint"
        project_arg = args.lint
    elif args.format is not None:
        action = "format"
        project_arg = args.format
    elif args.typecheck is not None:
        action = "typecheck"
        project_arg = args.typecheck
    elif args.sync:
        action = "sync"
        project_arg = "__all__"
    elif getattr(args, "all") is not None:
        action = "all"
        project_arg = getattr(args, "all")

    # Show help if no action provided
    if action is None:
        parser.print_help()
        return 0

    # Parse project
    target_projects = parse_project(None if project_arg == "__all__" else project_arg)

    print()
    print(f"{Colors.MAGENTA}{Colors.BOLD}signalfoundry Build Script{Colors.RESET}")
    print(f"Action: {action}")
    print(f"Target: {target_projects[0] if len(target_projects) == 1 else 'all projects'}")

    # Execute action
    actions_map = {
        "clean": lambda: action_clean(target_projects),
        "test": lambda: action_test(target_projects, args.no_coverage),
        "build": lambda: action_build(target_projects),
        "lint": lambda: action_lint(target_projects),
        "format": lambda: action_format(target_projects),
        "typecheck": lambda: action_typecheck(target_projects),
        "sync": action_sync,
        "all": lambda: action_all(target_projects, args.no_coverage),
    }

    success = actions_map[action]()
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


