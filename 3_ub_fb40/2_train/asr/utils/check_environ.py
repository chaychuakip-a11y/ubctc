import os
import datetime
import getpass
from subprocess import PIPE, Popen

try:
    import torch
    assert torch.__version__ == "1.7.0" or torch.__version__ == "1.6.0"
except Exception:
    raise RuntimeError("torch 1.6.0 or above version not found")

# get anaconda path
python_path = os.popen("which python").readline().strip()
if "envs" in python_path:
    anaconda_env = os.path.split(os.path.split(os.path.split(python_path)[0])[0])[1]
    anaconda_path = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(python_path)[0])[0])[0])[0], "bin")
    conda_path = os.path.join(os.path.split(anaconda_path)[0], "condabin")
    anaconda_lib = os.path.join(os.path.split(os.path.split(python_path)[0])[0], "lib")
else:
    anaconda_env = None
    anaconda_path = os.path.split(python_path)[0]
    conda_path = os.path.join(os.path.split(anaconda_path)[0], "condabin")
    anaconda_lib = os.path.join(os.path.split(anaconda_path)[0], "lib")

# sh path
source_sh_path = os.path.join("pytorch_env.bashrc")


# process for $PATH
PATH=[]
PATH.append(anaconda_path)
PATH.append("/opt/lib/cuda-10.2/bin")
PATH.append("/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/bin")
PATH.append("/opt/compiler/gcc-7.3.0-os7.2/bin")
PATH.append("/usr/local/bin")
PATH.append("/usr/bin")
PATH.append("/usr/local/sbin")
PATH.append("/usr/sbin")
PATH.append("/sbin")
PATH.append("/opt/ibutils/bin")
PATH.append(conda_path)

# process for $CPATH
CPATH = []
CPATH.append("/opt/lib/cuda-10.2/include")
CPATH.append("/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/include")

# process for LD_LIBRARY_PATH
LD_LIBRARY_PATH = []
LD_LIBRARY_PATH.append("/opt/lib/cuda-10.2/lib64")
LD_LIBRARY_PATH.append("/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64")
LD_LIBRARY_PATH.append("/opt/compiler/gcc-7.3.0-os7.2/lib64")
LD_LIBRARY_PATH.append("/opt/lib")
LD_LIBRARY_PATH.append("/opt/ufm/opensm/lib")
LD_LIBRARY_PATH.append(anaconda_lib)

# process for LIBRARY_PATH
LIBRARY_PATH = []
LIBRARY_PATH.append("/opt/lib/cuda-10.2/lib64")
LIBRARY_PATH.append("/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64")
LIBRARY_PATH.append("/opt/compiler/gcc-7.3.0-os7.2/lib64")
LIBRARY_PATH.append(anaconda_lib)

with open(source_sh_path, 'w') as f:
    f.write("export PATH={}\n".format(':'.join(PATH)))
    f.write("export CPATH={}\n".format(':'.join(CPATH)))
    f.write("export LD_LIBRARY_PATH={}\n".format(':'.join(LD_LIBRARY_PATH)))
    f.write("export LIBRARY_PATH={}\n".format(':'.join(LIBRARY_PATH)))
    f.write("export CUDA_HOME={}\n".format("/opt/lib/cuda-10.2"))
    f.write("source activate base\n")
    f.write("source activate {}\n".format(anaconda_env))





