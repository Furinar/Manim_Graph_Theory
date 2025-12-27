#!/usr/bin/env bash
set -euo pipefail

# Render the comparison video at low quality (quick preview)
./venv/bin/manim -ql algorithm_video.py AlgorithmComparisonVideo

# For full quality, uncomment the line below
# ./venv/bin/manim -pqh algorithm_video.py AlgorithmComparisonVideo
