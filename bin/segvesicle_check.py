#!/usr/bin/env python3
import sys
import os
try:
    import yaml
except ImportError:
    print("PyYAML is required for dependency checking. Please install it using 'pip install pyyaml'.")
    sys.exit(1)
    
import re
import fire
from packaging import version
from packaging.specifiers import SpecifierSet
import importlib.metadata    

def normalize_version_constraint(version_constraint):
    """
    Normalize version constraints to valid specifier format.
    """
    if version_constraint.startswith('=') and not version_constraint.startswith('=='):
        return '==' + version_constraint[1:]
    if version_constraint.endswith('.'):
        return version_constraint.rstrip('.')
    return version_constraint


def check_dependencies():
    """
    Check if all required pip dependencies are installed, ignoring other parts of environment.yml.
    """
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environment.yml')
    if not os.path.exists(env_file):
        print(f"Environment file not found at {env_file}. Cannot perform dependency check.")
        sys.exit(1)

    with open(env_file, 'r') as file:
        try:
            env = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error parsing environment.yml: {e}")
            sys.exit(1)

    # Focus only on pip dependencies
    pip_dependencies = []
    for dep in env.get('dependencies', []):
        if isinstance(dep, dict) and 'pip' in dep:
            pip_dependencies.extend(dep['pip'])

    missing = []
    incompatible = []

    for dep in pip_dependencies:
        # Parse package name and version constraint
        match = re.match(r"([a-zA-Z0-9_\-]+)([<>=!~]+[\d\.]+)?", dep)
        if not match:
            print(f"Invalid dependency format: {dep}")
            continue

        pkg = match.group(1)
        version_constraint = match.group(2)

        # Normalize version constraint
        if version_constraint:
            version_constraint = normalize_version_constraint(version_constraint)

        try:
            # Use importlib.metadata to get the installed version
            installed_version = importlib.metadata.version(pkg)
            if version_constraint:
                specifier = SpecifierSet(version_constraint)
                if not specifier.contains(installed_version):
                    incompatible.append((pkg, installed_version, version_constraint))
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)
        except Exception as e:
            print(f"Error checking {pkg}: {e}")

    # Report results
    if not missing and not incompatible:
        print("\nAll dependencies are installed and meet version requirements.\n")
        return

    print("\nDependency Check Failed:\n")
    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
    else:
        print("No missing packages detected.")

    if incompatible:
        print("\nWarning: Some packages have incompatible versions. This may cause unpredictable errors but might not affect software functionality.")
        print("Incompatible packages:")
        for pkg, installed, required in incompatible:
            print(f"  - {pkg}: Installed {installed}, Required {required}")
    else:
        print("No incompatible packages detected.")

    sys.exit(1)
    
if __name__ == '__main__':
    fire.Fire(check_dependencies)