# Recorded environments

Saved-result auditing uses only the Python 3.10+ standard library.

| Stage | Recorded environment |
|---|---|
| Original tree experiments | Python 3.11.13, arm64 CPU, two threads/fit; exact recorded major packages in requirements-cpu-recorded.txt |
| Supplement/GFP tree fits | Remote Linux x86_64 CPU, two threads/fit; variable fold concurrency, up to four for GFP |
| MLP and GraphSAGE | Python 3.12, PyTorch 2.5.1, NumPy 1.26.4, CUDA runtime 11.8, one shared RTX4090; float32 numerical rules in each protocol |
| GFP feature construction | Snap ML 1.17.2; supported native wheel and native-output validation required |
| Figures | LaTeX/PGFPlots/TikZ; Python figure generators; local Arial availability for the schematic figure sources |

The small requirements files distinguish recorded pins from additional
dependencies whose exact versions are not fully captured in these protocol
summaries. They are not an independently resolved universal lockfile. GPU and
CPU NumPy pins belong to different environments and should not be installed
together. Native-library/OS changes can alter exact score or timing bytes.

The repository supplies no environment binaries, package wheels, CUDA runtime or
third-party source code. Follow the package vendors' installation instructions
for a compatible platform, then register and verify a fresh scientific run.
