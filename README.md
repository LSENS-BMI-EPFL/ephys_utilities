# ephys_utilities

Utilities to process extracellular ephys data.
*Work in progress*, but already usable for NWB files and data loading and ephys metadata processing.


### Install

Install it in any environment. Two options:

Local clone + editable install (best for active development, since edits are picked up immediately without reinstalling):
```bash
git clone https://github.com/LSENS-BMI-EPFL/ephys_utilities.git
cd ephys_utilities
uv pip install -e .
# or: pip install -e .
```
Directly from GitHub (no local clone needed, good for HPC/other machines just consuming it):
```bash
uv pip install "git+https://github.com/LSENS-BMI-EPFL/ephys_utilities.git"
```

Pin to a branch/tag/commit if you want reproducibility:

```bash
uv pip install "git+https://github.com/LSENS-BMI-EPFL/ephys_utilities.git@main"
```

After that, import allen_utils, import neural_utils, etc. work from any script in that env, same as any other installed package.


### Contribute

If you'd like to contribute to this project, please follow the standard GitHub workflow:

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes and commit them
4. Push your changes to your fork
5. Create a pull request
