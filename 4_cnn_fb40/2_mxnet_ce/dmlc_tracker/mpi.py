"""
DMLC submission script, MPI version
"""
# pylint: disable=invalid-name
from __future__ import absolute_import

import subprocess, logging
from threading import Thread
from . import tracker

def get_mpi_env(envs):
    """get the mpirun command for setting the envornment
    support both openmpi and mpich2
    """
    """
    # decide MPI version.
    (_, err) = subprocess.Popen('mpirun',
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()
    cmd = ''
    if 'Open MPI' in err:
        for k, v in envs.items():
            cmd += ' -x %s=%s' % (k, str(v))
    elif 'mpich' in err:
        for k, v in envs.items():
            cmd += ' -env %s %s' % (k, str(v))
    else:
        raise RuntimeError('Unknown MPI Version')
    """

    cmd = ''
    for k, v in envs.items():
        cmd += ' -env %s %s' % (k, str(v))
    return cmd


def submit(args):
    """Submission script with MPI."""
    def mpi_submit(nworker, nserver, pass_envs):
        """Internal closure for job submission."""
        def run(prog):
            """run the program"""
            subprocess.check_call(prog, shell=True)

        cmd = ''
        if args.host_file is not None:
            cmd = '--hostfile %s ' % (args.host_file)
        cmd += ' ' + ' '.join(args.command)

        pass_envs['DMLC_JOB_CLUSTER'] = 'mpi'
        pass_envs['MXNET_CUDA_ALLOW_TENSOR_CORE'] = '0'
        pass_envs['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
        #pass_envs['DMLC_INTERFACE'] = 'eno2.100'
        #pass_envs['DMLC_INTERFACE'] = 'enp175s0.100'
        #pass_envs['DMLC_INTERFACE'] = 'enp59s0.100'
        pass_envs['DMLC_INTERFACE'] = 'eno2.100'
        
        print('DMLC_PS_ROOT_URI: '+pass_envs['DMLC_PS_ROOT_URI'])
        try:
            f = open('/etc/mpi/hostfile', mode='r')
            content = f.read()
            f.close()
            print(content)
        except Exception as ex:
            print(ex, '/etc/mpi/hostfile')
        # start workers
        if nworker > 0:
            logging.info('Start %d workers by mpirun' % nworker)
            pass_envs['DMLC_ROLE'] = 'worker'
            prog = 'mpirun -launcher rsh -launcher-exec /etc/mpi/kubexec.sh --hostfile /etc/mpi/hostfile -n %d %s %s' % (nworker, get_mpi_env(pass_envs), cmd)
            #prog = 'mpirun -n %d %s %s' % (nworker, get_mpi_env(pass_envs), cmd)
            print "worker: ", prog
            thread = Thread(target=run, args=(prog,))
            thread.setDaemon(True)
            thread.start()

        # start servers
        if nserver > 0:
            logging.info('Start %d servers by mpirun' % nserver)
            pass_envs['DMLC_ROLE'] = 'server'
            prog = 'mpirun -launcher rsh -launcher-exec /etc/mpi/kubexec.sh --hostfile /etc/mpi/hostfile -n %d %s %s' % (nserver, get_mpi_env(pass_envs), cmd)
            #prog = 'mpirun -n %d %s %s' % (nserver, get_mpi_env(pass_envs), cmd)
            print "server: ", prog
            thread = Thread(target=run, args=(prog,))
            thread.setDaemon(True)
            thread.start()


    tracker.submit(args.num_workers, args.num_servers,
                   fun_submit=mpi_submit,
                   pscmd=(' '.join(args.command)))
