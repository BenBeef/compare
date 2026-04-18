from typing import *
import pickle


class Node:
    def __init__(self, word: Union[str, int]) -> None:
        self.word = word
        self.cnt = 0
        self.chidren: Dict[Union[str, int], Node] = {}
        self.ordered_paths: List[List[str]] = []
    
    def add_words(self, words:List[Union[str, int]]):
        if not words:
            return 
            
        word = words[0]
        if word not in self.chidren:
            self.chidren[word] = Node(word)
        self.chidren[word].cnt += 1
        self.chidren[word].add_words(words[1:])
    

    def triversal(self, stack:List["Node"], results:List[List["Node"]]):
        # print(f'word:{self.word}, cnt:{self.cnt}')
        stack.append(self)
        if not self.chidren:
            results.append(stack.copy())

        for _, node in self.chidren.items():
            node.triversal(stack, results)
        
        stack.pop()
    
    def score(self, nodes:List["Node"])->Union[str, int]:
        nodes = nodes[1:]
        base = 1
        s = 0.0
        for i in range(len(nodes)-1, -1, -1):
            base = base * 1.0
            s += nodes[i].cnt * base
        return s

    def order(self, min_cnt=1):
        results:List[List[Node]] = []
        self.triversal([], results)
        results.sort(key=lambda x:self.score(x), reverse=True)
        self.ordered_paths = [[node.word for node in result[1:]] for result in results if result[-1].cnt >= min_cnt]

    def show_paths(self):
        self.order()
        print(f'-------------------------- path for:{self.word} --------------------------')
        for _path in self.ordered_paths:
            print(_path)

    def show(self):
        results:List[List[Node]] = []
        self.triversal([], results)
        for result in results:
            info = [f'{i}_{node.word}:{node.cnt}' for i, node in enumerate(result)]
            print(info)


class NGramMgr:

    def __init__(self, N=3, min_cnt=1) -> None:
        self.start_words:Dict[Union[str, int], Node] = {}
        self.N = N
        self.min_cnt = min_cnt  # 最低出现次数
        self.update = True
        self.word_2_hit: Dict[Union[str, int], List[int]] = {}
        self.total_offer = 0
        self.total_hit = 0
        self.collect = False
    
    def collect_hit(self, word, num, num_hit):
        num -= 1
        num_hit -= 1
        self.word_2_hit.setdefault(word, [0, 0])
        self.word_2_hit[word][0] += num
        self.word_2_hit[word][1] += num_hit
        self.total_hit += num_hit
        self.total_offer += num
    
    def show_hit(self):
        def _rate(x, y):
            return x / y if y else 0.0
        print('=======' * 4, 'n_gram_cache_hit', '=======' * 4)
        print(f'Total hit {self.total_hit} / {self.total_offer} = {_rate(self.total_hit, self.total_offer)}')

        word_hits = [(f'word:{word} hits:{hit_info[0]}/{hit_info[1]}', _rate(hit_info[0], hit_info[1])) for word, hit_info  in self.word_2_hit.items()]
        word_hits.sort(key=lambda x:-x[-1])
        for info, r in word_hits[:20]:
            print(f'{info} = {r}')
    
    def add_words(self, words:List[Union[str, int]], _order=True):
        if not self.update or not words or len(words) < self.N:
            return
        visited = set()
        for i in range(len(words)-self.N):
            group = words[i:i+self.N+1]
            word = group[0]
            if word not in self.start_words:
                self.start_words[word] = Node(word)
            self.start_words[word].add_words(group[1:])
            visited.add(word)
        
        if _order:
            for word in visited:
                self.start_words[word].order(min_cnt=self.min_cnt)

    def order_all(self):
        for word in self.start_words:
            self.start_words[word].order(min_cnt=self.min_cnt)

    def get_lookahead(self, word:Union[str, int], _all=False):
        if word not in self.start_words:
            if not _all:
                return None
            return None, []

        paths = self.start_words[word].ordered_paths
        if not paths:
            if not _all:
                return None
            return None, []
        
        if not _all:
            return paths[0]
        return paths[0], paths.copy()

    def show_all(self):
        for word, val in self.start_words.items():
            print(f'----------------- triversal info for:{word} ---------------------')
            val.show()
            val.show_paths()
            print('\n\n')
    

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read_from_pkl(cls, path, update=True):
        # Pickles may reference AICASGC_Season1.my_kernel... which is not importable
        # when the repo root is not on PYTHONPATH; resolve our classes locally.
        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "NGramMgr":
                    return cls
                if name == "Node":
                    return Node
                return super().find_class(module, name)

        with open(path, "rb") as f:
            obj = _U(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f"pickle at {path!r} is {type(obj).__name__}, expected {cls.__name__}")
        obj.update = update
        return obj
