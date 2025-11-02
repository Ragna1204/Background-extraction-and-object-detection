#!/usr/bin/env python3
"""
Test runner script for the motion detection project
"""

import sys
import subprocess
import os


def run_tests():
    """Run the test suite"""
    print("Running test suite...")

    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])

    # Run tests
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--tb=short",
        "--verbose"
    ], cwd=os.getcwd())

    return result.returncode


def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"Running {test_file}...")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "--tb=short",
        "--verbose"
    ], cwd=os.getcwd())

    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        exit_code = run_specific_test(test_file)
    else:
        # Run all tests
        exit_code = run_tests()

    sys.exit(exit_code)
