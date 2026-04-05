import os
import argparse
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="running model [local|dlp]")
parser.add_argument("-w", type=int, help="number of workers")
parser.add_argument("-g", type=int, help="number of gpus")
parser.add_argument("-l", type=str, help="stdout", default="stdout.log")
parser.add_argument("-o", type=str, help="stderr", default="stderr.log")
parser.add_argument("-d", type=str, help="description")
parser.add_argument("-n", type=str, help="name")
parser.add_argument("--config", type=str, help="config file")
parser.add_argument("-k", type=str, help="gpu kinds")
parser.add_argument("-r", type=str, help="reserved query name, will be public when not be assigned")
parser.add_argument("--proID", type=int, help="project ID")



if __name__ == "__main__":
    args = parser.parse_args()
    assert args.mode is not None, "must assign mode [dlp|local]"
    assert args.mode == "dlp" or args.mode == "local", "mode must be dlp or local"
    if args.mode == "dlp":
        assert args.g is not None, "must assign gpu number when use dlp mode"
    assert args.config is not None, "must assign config file"

    if args.mode == "dlp":
        assert args.w is not None, "when mode = dlp, must assign workers"
        assert args.k is not None, "when mode = dlp, must assign gpu kinds"

    

    if args.mode == "local":
        with open("train.py", 'w') as f:
            f.write("import sys\n")
            f.write("from asr.train import train\n")
            f.write("if __name__ == \"__main__\":\n")
            f.write("    mytrain = train()\n")
            f.write("    mytrain.load_config(sys.argv[1])\n")
            f.write("    mytrain.start_train()")

        with open("run_work.sh", 'w') as f:
            os.system("python -m asr.utils.check_environ")
            f.write("source {}\n".format(os.path.join(os.getcwd(), "pytorch_env.bashrc")))
            f.write("python train.py {}\n".format(args.config))
        os.system("bash run_work.sh")

    if args.mode == "dlp":
        with open("train.py", 'w') as f:
            f.write("import sys\n")
            f.write("from asr.train import train\n")
            f.write("if __name__ == \"__main__\":\n")
            f.write("    mytrain = train()\n")
            f.write("    mytrain.load_config(sys.argv[1])\n")
            f.write("    mytrain.start_train()")

        with open("run_work.sh", 'w') as f:
            os.system("python -m asr.utils.check_environ")
            f.write("source {}\n".format(os.path.join(os.getcwd(), "pytorch_env.bashrc")))
            if args.w > 1:
                f.write("export NCCL_SOCKET_IFNAME=eno2.100\n")
            f.write("python train.py {}\n".format(args.config))
        
        cmd = "/opt/dls_cli/dlp submit"
        cmd += " -a {}".format(getpass.getuser())
        cmd += " -d {}".format(args.d) if args.d is not None else ''
        cmd += " -n {}".format(args.n) if args.n is not None else ''
        cmd += " -l {}".format(args.l) if args.l is not None else ''
        cmd += " -o {}".format(args.o) if args.o is not None else ''
        cmd += " -i {}".format("reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2")
        cmd += " -g {}".format(args.g)
        cmd += " --useGpu"
        cmd += " -w {}".format(args.w)
        cmd += " -e {}".format("run_work.sh")
        if args.w > 1:
            cmd += " --useDist"
            cmd += " -b nccl2"
        cmd += " -t {}".format("PtJob")
        cmd += " -k {}".format(args.k)
        if args.r is not None:
            cmd += " -r {}".format(args.r)
        cmd += " --proID {}".format(args.proID)
        os.system(cmd)



    
