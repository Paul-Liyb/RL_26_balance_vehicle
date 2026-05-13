# MATLAB Reference

This folder contains the curated MATLAB reference file used by the Python simulation:

- `inverted_pendulum_on_self_balancing_robot.m`

The file was copied from the vendor WHEELTEC B585 DP2 materials under:

`WHEELTEC+B585+DP2+二阶平衡机器人附送资料/.../MATLAB程序/MATLAB程序/R2022a/inverted_pendulum_on_self_balancing_robot.m`

The Python implementation in `tools/lqr_from_matlab.py` reproduces the physical parameters, continuous matrices, discretization, and `dlqr` gain calculation from this script.

Keep this folder small. Do not commit the full vendor archive, PDFs, videos, APKs, or firmware bundles unless the team has explicitly decided that the repository should host those large/reference-only assets.
