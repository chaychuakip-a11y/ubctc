# develop by pcli2 2019 Nov.
# update by pcli2 2020 Aug.
# use python multiprocessing.managers to transmit information between workers
import multiprocessing
import time
import socket
from multiprocessing.managers import BaseManager
from threading import Barrier


class LogType():
    SYNC = 1
    COMMON = 2


class ProxyManager(BaseManager):
    pass


class Message():
    def __init__(self, host, port, world_size):
        self.host = host
        self.port = port
        self.world_size = world_size
        self.__is_server = False
        self.__logpath = None
        self.__log_queue = None
        self.__writer_process = None
        self.__barrier = None

    def start_server(self):
        self.__log_queue = multiprocessing.Queue(maxsize=500)
        self.__barrier = Barrier(self.world_size)
        ProxyManager.register("get_log_queue", callable=lambda: self.__log_queue)
        ProxyManager.register("get_barrier", callable=lambda: self.__barrier)
        self.__proxy_manager = ProxyManager(address=(socket.gethostbyname(self.host), self.port), authkey=b"key")
        self.__proxy_manager.start()
        self.__is_server = True

    def get_barrier(self):
        return self.__proxy_manager.get_barrier()

    def wait(self):
        MAX_RETRY = 5
        retry = 0
        while retry < MAX_RETRY:
            try:
                self.get_barrier().wait()
                break
            except Exception:
                time.sleep((retry + 1) * (retry + 1))
                retry = retry + 1
                continue
            raise RuntimeError("wait() achieved maximum failure times")



    def start_client(self):
        MAX_RETRY = 5
        ProxyManager.register("get_log_queue")
        ProxyManager.register("get_barrier")
        print(self.host)
        #self.__proxy_manager = ProxyManager(address=(socket.gethostbyname(self.host), self.port), authkey=b"key")
        self.__proxy_manager = ProxyManager(address=(self.host, self.port), authkey=b"key")
        retry = 0
        while retry < MAX_RETRY:
            try:
                self.__proxy_manager.connect()
                self.__log_queue = self.__proxy_manager.get_log_queue()
                break
            except Exception:
                time.sleep((retry + 1) * (retry + 1))
                retry = retry + 1
                continue
            raise RuntimeError("client started unsuccessfully")

    def set_logpath(self, logpath, NGpu=1):
        if self.__writer_process is not None:
            self.__stop_log_writer()
        self.__logpath = logpath
        self.__start_log_writer(NGpu)

    def __logwriter(self, NGpu):
        assert self.__logpath is not None, "logpath is not set"
        log_queue = self.__proxy_manager.get_log_queue()
        sync_dict = {}
        while True:
            hd = open(self.__logpath, 'a')
            data = log_queue.get()
            if data == "END":
                break
            else:
                data, rank, logtype, syncid, syncheader = data
            if logtype == LogType.SYNC:
                if syncid not in sync_dict:
                    sync_dict[syncid] = {}
                sync_dict[syncid][rank] = data
                if len(sync_dict[syncid]) == NGpu:
                    current_t = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
                    prefix = get_log_header(syncheader)
                    hd.write(prefix)
                    hd.write(current_t)
                    hd.write('\n')
                    keys = sync_dict[syncid].keys()
                    sorted_keys = sorted(keys, key=lambda key: int(key))
                    for key in sorted_keys:
                        hd.write(sync_dict[syncid][key] + '\n')
                    del sync_dict[syncid]
                else:
                    continue
            if logtype == LogType.COMMON:
                current_t = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
                if '\n' in data:
                    hd.write(current_t)
                    hd.write('\n')
                    hd.write(data)
                    hd.write('\n')
                else:
                    hd.write("{0} {1}\n".format(current_t, data))
            hd.close()

    def __start_log_writer(self, NGpu):
        assert self.__writer_process is None, "log writer process has already been started"
        self.__writer_process = multiprocessing.Process(target=self.__logwriter, args=(NGpu,))
        self.__writer_process.start()

    def __stop_log_writer(self):
        if (self.__writer_process is not None):
            if self.__writer_process.is_alive():
                self.__proxy_manager.get_log_queue().put("END")
                self.__writer_process.join()
                self.__writer_process = None
            else:
                self.__writer_process = None

    def getlog(self):
        return Log(self.__proxy_manager.get_log_queue())


    def shutdown(self):
        self.__stop_log_writer()
        if hasattr(self.__proxy_manager, "shutdown"):
            self.__proxy_manager.shutdown()


class Log():
    def __init__(self, log_queue):
        self.__log_queue = log_queue

    def log(self, data, rank=None, logtype=LogType.COMMON, syncid=None, syncheader=None):
        assert self.__log_queue is not None, "server or client is not started"
        if logtype == LogType.SYNC:
            assert syncid != None, "use sync log, but syncid is not specified"
            assert rank != None, "use sync log, but rank is not specified"
        self.__log_queue.put([data, rank, logtype, syncid, syncheader])


def get_log_header(log):
    log_len = len(log)
    prefix_len = int((90 - log_len) / 2)
    suffix_len = int((90 - log_len) - prefix_len)
    log = prefix_len * '*' + log + suffix_len * '*'
    log = log + '\n'
    return log
