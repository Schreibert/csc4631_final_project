from setuptools import setup, find_packages
import os
import subprocess
import sys

class CMakeBuild:
    """Helper to build C++ extension via CMake"""

    @staticmethod
    def build():
        # Build directory
        build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake
        subprocess.check_call([
            'cmake',
            '-DBUILD_TESTS=OFF',
            '-DBUILD_PYTHON_BINDINGS=ON',
            '..'
        ], cwd=build_dir)

        # Build
        subprocess.check_call([
            'cmake',
            '--build', '.',
            '--config', 'Release'
        ], cwd=build_dir)

# Try to build C++ module
try:
    CMakeBuild.build()
except Exception as e:
    print(f"Warning: Could not build C++ extension: {e}")
    print("You may need to build manually with CMake")

setup(
    name="balatro-env",
    version="0.1.0",
    description="Balatro poker simulator for reinforcement learning",
    packages=find_packages(),
    package_data={
        'balatro_env': ['_balatro_core.*'],  # Include compiled module
    },
    install_requires=[
        'numpy>=1.21.0',
        'gymnasium>=0.28.0',
    ],
    python_requires='>=3.8',
)
