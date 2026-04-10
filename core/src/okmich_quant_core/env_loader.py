import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import argparse


class EnvLoader:
    """
    Load environment configuration for different brokers and account types.

    Naming convention for env files:
    - .env.{broker}.{account_type}
    - Examples: .env.deriv.demo, .env.icmarkets.live
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def load(self, broker: str, account_type: str = "demo", override: bool = True) -> bool:
        env_file = self.base_dir / f".env.{broker}.{account_type}"

        if not env_file.exists():
            print(f"Warning: Environment file not found: {env_file}")
            return False

        print(f"Loading environment from: {env_file}")
        load_dotenv(env_file, override=override)

        # Validate required variables
        required_vars = ["TERMINAL_PATH", "LOGIN_ID", "LOGIN_PASSWORD", "LOGIN_SERVER"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(f"Error: Missing required environment variables: {missing_vars}")
            return False

        print(f"✓ Loaded {broker} {account_type} configuration")
        print(f"  - Server: {os.getenv('LOGIN_SERVER')}")
        print(f"  - Login ID: {os.getenv('LOGIN_ID')}")

        return True

    def load_from_env_var(self, env_var: str = "MT5_ENV", override: bool = True) -> bool:
        env_value = os.getenv(env_var)

        if not env_value:
            print(f"Warning: Environment variable {env_var} not set")
            return False

        parts = env_value.split(".")
        if len(parts) != 2:
            print(f"Error: Invalid format for {env_var}. Expected 'broker.account_type', got '{env_value}'")
            return False

        broker, account_type = parts
        return self.load(broker, account_type, override)

    def load_from_args(self, override: bool = True) -> bool:
        parser = argparse.ArgumentParser(description="MT5 Trading Script")
        parser.add_argument(
            "--broker",
            type=str,
            required=True,
            help="Broker name (e.g., deriv, icmarkets)",
        )
        parser.add_argument(
            "--account",
            type=str,
            default="demo",
            choices=["demo", "live"],
            help="Account type (demo or live)",
        )

        args, _ = parser.parse_known_args()
        return self.load(args.broker, args.account, override)

    def list_available_configs(self) -> list:
        env_files = list(self.base_dir.glob(".env.*.*"))
        configs = []

        for env_file in env_files:
            # Use .name (full filename) not .stem — Path.stem strips only the
            # last suffix, so ".env.deriv.demo".stem == ".env.deriv", giving
            # wrong token positions.  Parsing the full name produces 4 parts:
            # ['', 'env', 'broker', 'account_type'].
            parts = env_file.name.split(".")
            if len(parts) == 4 and parts[0] == "" and parts[1] == "env":
                configs.append((parts[2], parts[3]))

        return sorted(configs)

    def print_available_configs(self):
        """Print all available configurations."""
        configs = self.list_available_configs()

        if not configs:
            print("No environment configurations found.")
            return

        print("\nAvailable configurations:")
        print("-" * 40)
        for broker, account_type in configs:
            env_file = self.base_dir / f".env.{broker}.{account_type}"
            print(f"  • {broker:15} {account_type:6}  ({env_file.name})")
        print("-" * 40)


def load_broker_env(
    broker: str = None,
    account_type: str = "demo",
    base_dir: str = None,
    auto: bool = True,
) -> bool:
    """
    Convenient function to load broker environment.

    Args:
        broker: Broker name. If None, tries to load from command line args or env var
        account_type: Account type ('demo' or 'live')
        base_dir: Base directory for .env files
        auto: If True, automatically tries command line args, then env var

    Returns:
        True if loaded successfully, False otherwise

    Examples:
        # Explicit broker/account
        load_broker_env('deriv', 'demo')

        # Auto-detect from command line
        # python script.py --broker icmarkets --account live
        load_broker_env()

        # From environment variable
        # export MT5_ENV=deriv.demo
        load_broker_env()
    """
    loader = EnvLoader(base_dir)

    if broker:
        # Explicit broker specified
        return loader.load(broker, account_type)

    if auto:
        # Try command line arguments first
        if "--broker" in sys.argv:
            return loader.load_from_args()

        # Try environment variable
        if os.getenv("MT5_ENV"):
            return loader.load_from_env_var()

    # Fallback: list available models_research_configs and raise error
    print("\nError: No broker configuration specified.")
    loader.print_available_configs()
    print("\nPlease specify broker configuration using one of:")
    print("  1. Command line: python script.py --broker deriv --account demo")
    print("  2. Environment variable: export MT5_ENV=deriv.demo")
    print("  3. Code: load_broker_env('deriv', 'demo')")

    return False
