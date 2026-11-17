from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Clean up newline characters
        requirements = [req.replace("\n", "") for req in requirements]
        # Remove the local editable flag if present
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="f1_safety_car_predictor",
    version="0.0.1",
    author="Kanabe Oshobugie Sadiq",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)