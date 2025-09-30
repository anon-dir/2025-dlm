# Denoising language model (DLM)

This repo provides a full setup for DLMs.

The setup is based on:

* [Sisyphus workflow manager](https://github.com/rwth-i6/sisyphus).
  Job classes define individual workflow steps, such as training a model, or evaluating a model,
  or preparing data, etc.
  Sisyphus manages dependencies between jobs, and can run jobs locally or on a cluster (e.g. via [Slurm](https://github.com/SchedMD/slurm)).
* Existing Sisyphus recipes / job classes from
  [i6_core](https://github.com/rwth-i6/i6_core),
  [i6_experiments](https://github.com/rwth-i6/i6_experiments).
* [RETURNN](https://github.com/rwth-i6/returnn).
  The main deep learning framework used in this setup,
  based on [PyTorch](https://pytorch.org/),
  providing `Tensor` with named dimensions (`Dim` objects),
  and many high-level modules for Transformers and other neural network architectures.
  We derive our code (modeling, training, decoding, recipes) from existing example code.


## Usage

In the setup directory (see setup below), run:

```shell
python ./sis manager recipe/denoising_lm/sis_recipe/<recipe_name>.py
```

where `<recipe_name>` is one of the recipes provided in `src/denoising_lm/sis_recipe/`.

## Setup

1. Create the new setup folder in your user directory, typically like `~/setups/<setup_name>` or `~/experiments/<setup_name>`. This will be your new setup root directory.
```
mkdir ~/experiments/<setup_name>
cd ~/experiments/<setup_name>
```

Optional: The setup directory should be a Git repo itself,
to keep track of the changes. You can do now:

```
git init .
edit README.md  # write a short description about your setup
git add README.md
git commit . -m initial

# some initial content for gitignore
cat << EOF > .gitignore
/output
/alias
/tools
/recipe
.*.swp
*.pyc
__pycache__
.idea
*.history*
.directory
EOF
git add .gitignore
git commit .gitignore -m gitignore
```

2. Create a new work folder under a "work" file system and link this as `work` into the Sisyphus setup root (`~/experiments/<setup_name>`).
```
mkdir -p /work/<project>/<username>/sisyphus_work_dirs/<setup_name>
ln -s /work/<project>/<username>/sisyphus_work_dirs/<setup_name> work
```

3. Create a recipe folder in the Sisyphus setup root (`~/experiments/<setup_name>`) and clone the necessary recipe repositories:
```
mkdir recipe

git clone git@github.com:rwth-i6/i6_core.git recipe/i6_core
git clone git@github.com:rwth-i6/i6_experiments.git recipe/i6_experiments
git clone git@github.com:rwth-i6/returnn_common.git recipe/returnn_common

mkdir repos
git clone <denoising-lm-repo-url> repos/denoising_lm
ln -s repos/denoising_lm/src/denoising_lm recipe/denoising_lm
```

If the access is denied for the Github repositories, you need to add your public ssh key (usually `~/.ssh/id_rsa.pub`) to your Github account.
This can be done by pasting the content (displayed with `cat ~/.ssh/id_rsa.pub`) into your Github key settings.
More information on adding keys to a Github account can be found [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

4. Setup Sisyphus and other setup-wide tools.

```shell
mkdir tools

git clone git@github.com:rwth-i6/sisyphus.git tools/sisyphus
ln -s tools/sisyphus/sis

git clone git@github.com:rwth-i6/returnn.git tools/returnn
```

5. Add a `settings.py` with Sisyphus settings. For example:

```python
import os
import sys

_root_dir = os.path.dirname(os.path.abspath(__file__))

RETURNN_PYTHON_EXE = sys.executable
RETURNN_ROOT = _root_dir + "/tools/returnn"
sys.path.insert(0, RETURNN_ROOT)

VERBOSE_TRACEBACK_TYPE = 'better_exchook'
USE_SIGNAL_HANDLERS = True

def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.simple_linux_utility_for_resource_management_engine import (
        SimpleLinuxUtilityForResourceManagementEngine,  # Slurm
    )
    # Other engines are available, see sisyphus documentation.

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=8),
            "local": LocalEngine(cpus=8, gpus=1),
            "slurm": SimpleLinuxUtilityForResourceManagementEngine(
                default_rqmt={"cpu": 4, "mem": "1G", "gpu": 0, "time": 1}
            ),
        },
        default_engine="local",  # or "slurm" ...
    )

```

You should adapt this file.
This file is loaded via sisyphus.global_settings.py,
update_global_settings_from_file specifically.
See [Sisyphus documentation](https://sisyphus-workflow-manager.readthedocs.io/).

6. Create some Python environment.
You can use `virtualenv` or `conda` or other tools.
We tested our setup with PyTorch 2.5.1.
Install missing dependencies.
