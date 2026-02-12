"""
Utility to load prompts from R2E-Gym YAML configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional


def get_r2egym_config_path(scaffold: str) -> Path:
    """
    Get the path to R2E-Gym config directory.
    
    Args:
        scaffold: Scaffold type ('r2egym', 'sweagent', 'openhands')
    
    Returns:
        Path to the R2E-Gym config directory for the specified scaffold
    """
    # First try to import r2egym and find its installation path
    try:
        import r2egym
        r2egym_path = Path(r2egym.__file__).parent
        config_path = r2egym_path / f"agenthub/config/{scaffold}"
        if config_path.exists():
            return config_path
    except ImportError:
        pass
    
    # Fallback: check environment variable
    if "R2EGYM_CONFIG_PATH" in os.environ:
        config_path = Path(os.environ["R2EGYM_CONFIG_PATH"])
        if config_path.exists():
            return config_path
    
    raise FileNotFoundError(
        f"Could not find R2E-Gym config directory for scaffold '{scaffold}'. "
        "Please ensure r2egym is installed or set the R2EGYM_CONFIG_PATH environment variable."
    )


# def load_prompts_from_yaml(domain: str, scaffold: str, version: str = "") -> Tuple[str, str]:
#     """
#     Load system_prompt and instance_prompt from R2E-Gym YAML config.
    
#     Args:
#         domain: Domain name (e.g., 'math_solver', 'physics_solver', 'chem_solver')
#         scaffold: Scaffold type ('r2egym', 'sweagent', 'openhands')
#         version: Optional version suffix (e.g., '_v1', '_v2')
    
#     Returns:
#         Tuple of (system_prompt, instance_prompt)
    
#     Examples:
#         >>> system_prompt, user_prompt = load_prompts_from_yaml('math_solver', 'openhands')
#         >>> system_prompt, user_prompt = load_prompts_from_yaml('math_solver', 'openhands', '_v2')
#         >>> system_prompt, user_prompt = load_prompts_from_yaml('math_solver', 'sweagent')
#     """
#     config_dir = get_r2egym_config_path(scaffold=scaffold)
#     yaml_file = config_dir / f"{domain}{version}.yaml"
    
#     if not yaml_file.exists():
#         raise FileNotFoundError(f"Config file not found: {yaml_file}")
    
#     with open(yaml_file, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)

#     system_prompt = config['system_prompt']
#     instance_prompt = config['instance_prompt']
    
    
#     return system_prompt, instance_prompt


# def get_available_domains(scaffold: str) -> Dict[str, list]:
#     """
#     Get a list of all available domain configurations.
    
#     Args:
#         scaffold: Scaffold type ('r2egym', 'sweagent', 'openhands')
    
#     Returns:
#         Dictionary mapping domain base names to their available versions
#     """
#     try:
#         config_dir = get_r2egym_config_path(scaffold=scaffold)
#     except FileNotFoundError:
#         return {}
    
#     domains = {}
#     for yaml_file in config_dir.glob("*.yaml"):
#         name = yaml_file.stem
        
#         # Extract base domain name and version
#         if '_v' in name:
#             base_name = name[:name.rfind('_v')]
#             version = name[name.rfind('_v'):]
#         else:
#             base_name = name
#             version = ''
        
#         if base_name not in domains:
#             domains[base_name] = []
#         domains[base_name].append(version)
    
#     return domains

# 没用上
# # Pre-load common domain prompts for convenience
# def get_math_solver_prompts(scaffold: str, version: str = "") -> Tuple[str, str]:
#     """Load math solver prompts."""
#     return load_prompts_from_yaml('math_solver', scaffold=scaffold, version=version)


# def get_physics_solver_prompts(scaffold: str, version: str = "") -> Tuple[str, str]:
#     """Load physics solver prompts."""
#     return load_prompts_from_yaml('physics_solver', scaffold=scaffold, version=version)


# def get_chem_solver_prompts(scaffold: str, version: str = "") -> Tuple[str, str]:
#     """Load chemistry solver prompts."""
#     return load_prompts_from_yaml('chem_solver', scaffold=scaffold, version=version)


# def get_ugphysics_solver_prompts(scaffold: str, version: str = "") -> Tuple[str, str]:
#     """Load UG physics solver prompts."""
#     return load_prompts_from_yaml('ugphysics_solver', scaffold=scaffold, version=version)


# def get_medxpertqa_solver_prompts(scaffold: str, version: str = "") -> Tuple[str, str]:
#     """Load medical expert QA solver prompts."""
#     return load_prompts_from_yaml('medxpertqa_solver', scaffold=scaffold, version=version)


if __name__ == "__main__":
    # Test the loader
    print("Available domains:")
    for domain, versions in get_available_domains().items():
        print(f"  {domain}: {versions}")
    
    print("\n" + "="*80)
    print("Testing math_solver prompt loading:")
    print("="*80)
    
    try:
        system_prompt, instance_prompt = load_prompts_from_yaml('math_solver', scaffold='openhands', version="")
        print(f"\nSystem prompt preview (first 200 chars):")
        print(system_prompt[:200] + "...")
        print(f"\nInstance prompt preview (first 200 chars):")
        print(instance_prompt[:200] + "...")
    except Exception as e:
        print(f"Error: {e}")
