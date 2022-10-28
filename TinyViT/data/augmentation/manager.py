import os
import multiprocessing
import tempfile


class _Writer:
    def __init__(self, path, rank):
        self.msg_queue = multiprocessing.Queue()
        self.worker = multiprocessing.Process(
            target=self._async_manager_worker_fn,
            args=(self.msg_queue, path, rank),
        )
        self.worker.start()

    def write(self, key: str, value: bytes) -> bool:
        self.msg_queue.put((key, value))
        return True

    class _WORKER_MSG:
        KILL = 4

    def _async_manager_worker_fn(self, msg_queue, path, rank):
        # path: xxx/logits_top100_epoch0
        rank_name = f'rank{rank}'
        # logits_top100_epoch0_rank0
        basename = os.path.basename(path) + f'_{rank_name}'
        tmp_handle = tempfile.TemporaryDirectory(prefix='tinyvit_' + basename)

        # tmp_dir/tinyvit_logits_top100_epoch0_rank0
        temp_dirname = tmp_handle.name

        tmp_filename = os.path.join(temp_dirname, rank_name)
        # tmp_dir/tinyvit_logits_top100_epoch0_rank0/rank0-keys.txt
        keys_fname = tmp_filename + '-keys.txt'
        values_fname = tmp_filename + '-values.bin'

        keys_file = open(keys_fname, 'w')
        values_file = open(values_fname, 'wb')
        keys = dict()

        while 1:
            item = msg_queue.get()
            if item == _Writer._WORKER_MSG.KILL:
                break
            key, value = item
            if key in keys:
                continue
            idx = len(keys)
            keys[key] = idx
            keys_file.write(key + '\n')
            values_file.write(value)

        keys_file.close()
        values_file.close()

        os.makedirs(path, exist_ok=True)
        os.system(f'mv {temp_dirname}/* {path}/')
        print(f"Save logits over: {path}")

    def __del__(self):
        if self.worker is not None:
            self.msg_queue.put(_Writer._WORKER_MSG.KILL)
            self.worker.join()


class _Reader:
    def __init__(self, path: str, item_size: int, rank: int):
        self.rank = rank
        self.item_size = item_size
        self.packages = self.search_packages(path)
        self.packages_visited = [False] * len(self.packages)
        # key -> package idx
        self.keys = dict()

    def read(self, key: str) -> bytes:
        pkg_idx, value_idx = self.keys.get(key, (None, None))
        if pkg_idx is None:
            pkg_idx, value_idx = self.find_item_in_packages(key)
        return self.packages[pkg_idx][value_idx]

    def find_item_in_packages(self, key: str) -> (int, int):
        for pkg_idx, pkg in enumerate(self.packages):
            if not self.packages_visited[pkg_idx]:
                self.packages_visited[pkg_idx] = True
                # load keys
                keys_fname = pkg.name + '-keys.txt'
                with open(keys_fname, 'r') as keys_file:
                    for i, k in enumerate(keys_file.readlines()):
                        k = k.strip()
                        self.keys[k] = (pkg_idx, i)
                if key in self.keys:
                    return self.keys[key]
        raise KeyError(key)

    def search_packages(self, path):
        assert os.path.isdir(path), f'[Error] Reading logits fails. Path {path} not found.'
        names = self.search_packages_names(path)
        return [_Reader._PackageReader(name, self.item_size) for name in names]

    def search_packages_names(self, path):
        names = []
        VALUES_POSTFIX = '-values.bin'
        for name in os.listdir(path):
            if name.endswith(VALUES_POSTFIX):
                names.append(name[:-len(VALUES_POSTFIX)])

        num_packages = len(names)

        def rank_key_fn(name):
            r = int(name[4:])
            return (r - self.rank) % num_packages

        # move the rankx-keys.txt to the front
        names.sort(key=rank_key_fn)
        names = list(map(lambda x: os.path.join(path, x), names))
        return names

    class _PackageReader:
        def __init__(self, name, item_size):
            self.name = name
            self.item_size = item_size

            # delay to create handle
            self.values_file = None

        def __getitem__(self, idx: int):
            self._ensure_handle_created()
            self.values_file.seek(self.item_size * idx)
            return self.values_file.read(self.item_size)

        def _ensure_handle_created(self):
            if self.values_file is None:
                values_fname = self.name + '-values.bin'
                self.values_file = open(values_fname, 'rb')


class TxtManager:
    def __init__(self, path: str, item_size: int, rank: int):
        self.path = path
        self.writer = None
        self.reader = None
        self.item_size = item_size
        self.rank = rank

    def write(self, key: str, value: bytes) -> bool:
        if self.writer is None:
            self.writer = _Writer(self.path, self.rank)
        return self.writer.write(key, value)

    def read(self, key: str) -> bytes:
        if self.reader is None:
            self.reader = _Reader(self.path, self.item_size, self.rank)
        return self.reader.read(key)
