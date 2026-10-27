# setup.py
from setuptools import setup, find_packages

setup(
    name        = "aegis-slam",
    version     = "1.0.0",
    description = "Self-healing visual SLAM on OpenCV 5 with an observe-diagnose-act-verify agent supervisor",
    packages    = find_packages(include=["slam", "slam.*", "aegis", "aegis.*"]),
    install_requires=[
        "opencv-python-headless==5.0.0.93",
        "numpy",
        "scipy",
        "matplotlib",
        "lz4",
        "tqdm",
    ],
    extras_require={
        "service": ["fastapi", "uvicorn[standard]", "python-multipart"],
        "ba":      ["pyceres", "pycolmap"],
        "learned": ["torch", "lightglue @ git+https://github.com/cvg/LightGlue.git"],
        "aws":     ["boto3"],
    },
    entry_points={
        "console_scripts": [
            "aegis-slam=aegis.run:main",
            "aegis-slam-server=aegis.server:serve",
            "aegis-slam-eval=aegis.evaluate:main",
        ],
    },
)
