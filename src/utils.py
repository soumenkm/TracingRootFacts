import ast
import copy
import hashlib
import json
import os
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from datetime import datetime
from functools import wraps
from threading import Lock

import _pickle as pickle
import lang2vec.lang2vec as l2v
import pandas as pd
import torch
from langcodes import Language
from tqdm import tqdm

from constants import LOADING_FILES_LAMBDA, NEURONS_ROOT


def get_logger(file_name, name):
    import logging

    from constants import LOGGING_ROOT
    now = datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    day_month_year = '{}-{}-{}:{}'.format(year, month, day, hour)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        filename=os.path.join(LOGGING_ROOT, f"{file_name}-{day_month_year}"))
    logger = logging.getLogger(name)
    return logger


def batchify(sents, batch_size):
    l = len(sents)
    for ndx in range(0, l, batch_size):
        yield sents[ndx : min(ndx + batch_size, l)] # type: ignore


def parse_list(pred):
    return ast.literal_eval(pred)


def tokens2id(pred, tokenizer):
    token_ids = []
    for token in pred:
        assert len(token) == 1
        token_ids.append(tokenizer.convert_tokens_to_ids(token)[0])
    return token_ids


def strip_space(ids, is_wrapped=True):
    if is_wrapped:
        res = [_id[0] for _id in ids if _id[0] != 6]
    else:
        res = [_id for _id in ids if _id != 6]
    if len(res) > 0:
        assert isinstance(res[0], int)
    return res


def adding_tokenization_to_prediction(lang, rel, root):
    tgt_fn = os.path.join(root, lang, f"{lang}-{rel}.csv")
    if not os.path.exists(tgt_fn):
        return pd.DataFrame()
    df = pd.read_csv(tgt_fn)
    if "pred_ids" in df.columns:
        # print("The conversion is already done")
        return
    df["prediction"] = df["prediction"].apply(lambda x: parse_list(x))

    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    df["pred_ids"] = df["prediction"].apply(lambda x: tokens2id(x, tokenizer))
    if "Unnamed: 0.1" in df.columns:
        df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])
    df.to_csv(tgt_fn)
    print(f"Finished Converstion {lang}-{rel}. Rewrite {tgt_fn}")
    return df

def loader(func):
    def _define_dump_file_name(*arg, **kwargs):
        if func.__name__ not in LOADING_FILES_LAMBDA:
            raise NotImplementedError(
                "The loading file path for {} is not defined.".format(func.__name__)
            )
        if callable(LOADING_FILES_LAMBDA[func.__name__]):
            if "dataset" not in kwargs:
                if not arg or (arg and arg[0].__class__.__name__ != "MaskedDataset"):
                    raise NotImplementedError(
                        f"To use loader decorator for {func.__name__}, it must implement dataset as the argument."
                    )
                return LOADING_FILES_LAMBDA[func.__name__](arg[0])
                    
            if "lang" in kwargs and "rel" in kwargs:
                file_name = LOADING_FILES_LAMBDA[func.__name__](
                    kwargs["lang"], kwargs["rel"], kwargs["dataset"]
                )
            elif "lang" in kwargs:
                file_name = LOADING_FILES_LAMBDA[func.__name__](
                    kwargs["lang"], kwargs["dataset"]
                )
            else:
                file_name = LOADING_FILES_LAMBDA[func.__name__](kwargs["dataset"])
        else:
            file_name = LOADING_FILES_LAMBDA[func.__name__]
        return file_name

    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name = _define_dump_file_name(*args, **kwargs)
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        reload = kwargs["reload"] if "reload" in kwargs else False
        if os.path.exists(file_name) and reload == False:
            if verbose:
                print(
                    "Load pre-saved file {} for function {}".format(
                        file_name, func.__name__
                    )
                )
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        result = func(*args, **kwargs)
        with open(file_name, "wb") as fp:
            pickle.dump(result, fp)
        if verbose:
            print(
                "Successfully executed the function {} and dumped result to the file {}".format(
                    func.__name__, file_name
                )
            )
        return result

    return wrapper


def neuron_loader(func):
    def hashing(sent):
        return hashlib.sha256(sent.encode()).hexdigest()

    def check_argument(*args, **kwargs):
        if "sentences" not in kwargs:
            raise NotImplementedError("The sents argument must be provided to use <neuron_loader>")
    
    def concat(new_neurons, dumped_neurons, dumped_indexes):
        new_idx = 0
        dumped_idx = 0

        result = []
        for sign in dumped_indexes:
            if sign:
                result.append(dumped_neurons[dumped_idx])
                dumped_idx += 1
            else:
                result.append(new_neurons[new_idx])
                new_idx += 1
        return result


    @wraps(func)
    def wrapper(*args, **kwargs):
        check_argument(*args, **kwargs)
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        reload = kwargs["reload"] if "reload" in kwargs else False
        
        model_name = args[0].name
        sents = kwargs["sentences"]
        new_sents = copy.deepcopy(kwargs["sentences"])
        hashes = [hashing(sent) for sent in sents]
        dumped_index = [False for i in range(len(sents))]

        dumped_neurons = []
        for idx in range(len(sents)):
            file_name = os.path.join(NEURONS_ROOT[model_name], hashes[idx])
            if os.path.exists(file_name) and reload == False:
                if verbose:
                    print(f"Load pre-saved file {file_name} for function {func.__name__}")
                dumped_neurons.append(torch.load(file_name))
                dumped_index[idx] = True
                new_sents.remove(sents[idx])
        
        if all(dumped_index):
            return dumped_neurons
        
        kwargs["sentences"] = new_sents
        new_neurons = func(*args, **kwargs)
        assert(len(new_neurons) + len(dumped_neurons) == len(sents))
        
        result = concat(new_neurons, dumped_neurons, dumped_index)
        
        for idx, dump_sign in enumerate(dumped_index):
            if not dump_sign:
                file_name = os.path.join(NEURONS_ROOT[model_name], hashes[idx])
                torch.save(result[idx], file_name)
        if verbose:
            print(f"Successfully executed the function {func.__name__}")
        return result

    return wrapper


def prober_loader(func):
    def _define_dump_file_name(*arg, **kwargs):
        if func.__name__ not in LOADING_FILES_LAMBDA:
            raise NotImplementedError(
                "The loading file path for {} is not defined.".format(func.__name__)
            )
        
        assert(callable(LOADING_FILES_LAMBDA[func.__name__]))
        if "rel" not in kwargs or "probing_type" not in kwargs:
            raise NotImplementedError(f"The `rel` and `probing_type` must be arguement for prober_loader function {func.__name__}")
        
        return LOADING_FILES_LAMBDA[func.__name__](arg[0].model_name, arg[0].match_type, kwargs["probing_type"], kwargs["rel"])

    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name = _define_dump_file_name(*args, **kwargs)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        reload = kwargs["reload"] if "reload" in kwargs else False
        if os.path.exists(file_name) and reload == False:
            if verbose:
                print(
                    "Load pre-saved file {} for function {}".format(
                        file_name, func.__name__
                    )
                )
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        result = func(*args, **kwargs)
        with open(file_name, "wb") as fp:
            pickle.dump(result, fp)
        if verbose:
            print(
                "Successfully executed the function {} and dumped result to the file {}".format(
                    func.__name__, file_name
                )
            )
        return result

    return wrapper

@loader
def load_objects(lang, dataset, tokenizer=None, reload=False):
    PATH1 = "../datasets/TREx_multilingual_objects/"
    PATH2 = "../datasets/GoogleRE_objects"
    candidates = {}
    for root in [PATH1, PATH2]:
        object_path = os.path.join(root, f"{lang}.json")
        with open(object_path) as f:
            candidates.update(json.load(f))
    del candidates['date_of_birth']
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    objs_num_dict = {}
    for rel in candidates.keys():
        objs_num_dict[rel] = {}
        maxlen = 0
        for obj in candidates[rel]["objects"]:
            maxlen = max(maxlen, len(tokenizer.tokenize(obj)))
        for i in range(1, maxlen + 1):
            objs_num_dict[rel].update({i: {}})
        for obj in candidates[rel]["objects"]:
            tokens = tokenizer.tokenize(obj)
            obj_len = len(tokens)
            objs_num_dict[rel][obj_len].update(
                {obj: tokenizer.convert_tokens_to_ids(tokens)}
            )
    return objs_num_dict


@loader
def load_subjects(lang, dataset, tokenizer=None, reload=False):
    PATH1 = "../datasets/TREx_multilingual_objects/"
    PATH2 = "../datasets/GoogleRE_objects"
    candidates = {}
    for root in [PATH1, PATH2]:
        object_path = os.path.join(root, f"{lang}.json")
        with open(object_path) as f:
            candidates.update(json.load(f))
    
    with open(object_path) as f:
        candidates = json.load(f)
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    
    subs_num_dict = {}
    for rel in candidates.keys():
        subs_num_dict[rel] = {}
        maxlen = 0
        for sub in candidates[rel]["subjects"]:
            maxlen = max(maxlen, len(tokenizer.tokenize(sub)))
        for i in range(1, maxlen + 1):
            subs_num_dict[rel].update({i: {}})
        for sub in candidates[rel]["subjects"]:
            tokens = tokenizer.tokenize(sub)
            sub_len = len(tokens)
            subs_num_dict[rel][sub_len].update(
                {sub: tokenizer.convert_tokens_to_ids(tokens)}
            )
    return subs_num_dict


@loader
def language_distance(langs, reload=False):
    en_idx = langs.index("en")
    langs_alpha3 = [Language.get(lang).to_alpha3() for lang in langs]
    distances = {"syntactic": l2v.syntactic_distance(langs_alpha3)[en_idx]}
    distances["geographic"] = l2v.geographic_distance(langs_alpha3)[en_idx]
    distances["phonological"] = l2v.phonological_distance(langs_alpha3)[en_idx]
    distances["genetic"] = l2v.genetic_distance(langs_alpha3)[en_idx]
    distances["inventory"] = l2v.inventory_distance(langs_alpha3)[en_idx]
    distances["featural"] = l2v.featural_distance(langs_alpha3)[en_idx]
    return distances

def language_distance_matrix(langs):
    langs_alpha3 = [Language.get(lang).to_alpha3() for lang in langs]
    return (
        l2v.syntactic_distance(langs_alpha3)
        + l2v.geographic_distance(langs_alpha3)
        + l2v.phonological_distance(langs_alpha3)
        + l2v.genetic_distance(langs_alpha3)
        + l2v.inventory_distance(langs_alpha3)
        + l2v.featural_distance(langs_alpha3)
    )

class ResourcePool_2():
    def __init__(self, resources, param_iterator, parallel_type='cpu-extensive'):
        self.resources = resources
        thread_num = len(resources) 
        self.states = [0 for _ in range(thread_num)]
        if parallel_type == 'io-extensive':
            self.executor = ThreadPoolExecutor(max_workers=thread_num)
        elif parallel_type == 'cpu-extensive':
            self.executor = ProcessPoolExecutor(max_workers=thread_num)

        self.state_lock = Lock()
        self.param_iterator = param_iterator

    def assign(self):
        while True:
            for idx, state in enumerate(self.states):
                if state == 0:
                    with self.state_lock:
                        self.states[idx] = 1
                    return idx, self.resources[idx]
            time.sleep(0.02)

    def release(self, idx):
        with self.state_lock:
            self.states[idx] = 0

    def run(self):
        futures = [self.executor.submit(self.task, params) for params in self.param_iterator]
        for future in as_completed(futures):
            future.result()
            
            
    def task(self, args):
        print("111")
        res = self.assign()
        print(f"The {args}-th iteration, get resource {res}")
        time.sleep(1)
        self.release(res)

def split_list(input_list, num_splits):
    avg_chunk_size = len(input_list) // num_splits
    remainder = len(input_list) % num_splits

    split_lists = []
    start = 0

    for i in range(num_splits):
        chunk_size = avg_chunk_size + 1 if i < remainder else avg_chunk_size
        end = start + chunk_size
        split_lists.append(input_list[start:end])
        start = end

    return split_lists

def chunk_list(input_list, chunk_size):
    res = []
    for i in range(0, len(input_list), chunk_size):
        res.append(input_list[i:i+chunk_size])
    return res

def chunk_list_by_value_range(input_list: list, chunk_size: int, max_val=None) -> tuple[list[list], list[list], list[list]]:
    """ Chunkize list by range of values. The range is decided by chunk_size
    Args:
        input_list (list): target list with data
        chunk_size (int): the size of each chunk. 
        max_val (bool): To aviod the long-tail list, it put all values into one sub-list if is bigger than max_val. 
        descendent (bool, optional): _description_. Defaults to False.

    Returns:
        chunked_list (list[list]): A list of list where each sub-list contains values those fall into the same range. 
        chunked_range (list[list]): A list of list where each sub-list contains two values, the start index and end index for this sublist. 
        chunked_idx (list[list]): A list of list where each sub-list contains the index of each value in `chunked_list`
    E.g., 
        >>> chunked_list, chunked_range, chunked_idx = \
                chunk_list_by_value_range([1, 2, 3, 4, 5, 8, 9, 10, 12, 17, 100, 153]], chunk_size=5, max_val=15)
        >>> chunked_list
            [[1, 2, 3, 4], [5, 8, 9], [10, 12], [17, 100, 153]]
        >>> chunked_range
            [[0, 4], [5, 9], [10, 14], [15, 153]] 
        >>> chunked_idx
            [[0, 1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9], [9, 10, 11]]

    """
    if not max_val:
        max_val = max(input_list)
    input_list = sorted(input_list)
    chunked_list = []
    chunked_range = []
    chunked_idx = []    
    for i in range(0, max(input_list), chunk_size):
        if i >= max_val:
            chunked_range.append([i, max(input_list)])
            break
        chunked_range.append([i, i+chunk_size-1])
    
    start_idx = 0
    for start_val, end_val in chunked_range:
        chunked_list.append([])
        chunked_idx.append([])
        for j in range(start_idx, len(input_list)):
            chunked_idx[-1].append(j)
            if input_list[j] >= start_val and input_list[j] <= end_val:
                chunked_list[-1].append(input_list[j])
            else:
                start_idx = j
                break
    return chunked_list, chunked_range, chunked_idx

def run_parallel_with_resource_2(resources, param_iterator, func):
    import multiprocessing
    from collections import deque
    from multiprocessing import Queue, Value        
    param_queue = deque(param_iterator)

    with multiprocessing.Manager() as manager:
        resource_queue = manager.Queue(maxsize=len(resources))
        for resource in resources:
            resource_wrapper = manager.Value('i', resource)
            resource_queue.put(resource_wrapper)
            print(f"Inject resource: {resource_wrapper, resource_wrapper.get()}")
        
        with ProcessPoolExecutor(max_workers=len(resources)) as executor:
            while len(param_queue) != 0:
                if not resource_queue.empty():
                    resource_wrapper = resource_queue.get()
                    # print(f"Get resource: {resource}")
                    params = param_queue.pop()
                    future = executor.submit(func, params, resource_wrapper)
                    future.add_done_callback(lambda future: (resource_queue.put(resource_wrapper), print(f"Returned resource: {resource_wrapper}, returned queue size: {resource_queue.qsize()}")))
        while not resource_queue.empty():
            resource_wrapper = resource_queue.get()
            print(resource_wrapper, resource_wrapper.get())


def run_func_in_loop(func, params, resource):
    for param in params:
        func(param, resource)

def run_parallel_with_resource(resources, param_iterator, func):
    params_lists = split_list(list(param_iterator), len(resources))
    with ProcessPoolExecutor(max_workers=len(resources)) as executor:
        futures = []
        for i, params in enumerate(params_lists):
            resource = resources[i]
            futures.append(executor.submit(run_func_in_loop, func, params, resource))
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    import time
    def test_func(params, resource):
        print(f"params: {params}, resource: {resource}")
        time.sleep(1)

    resources = [f'model-{i}' for i in list(range(10))]
    resources = range(10)
    func_param_iterator = [f'parameter-{i}' for i in list(range(50))]
    run_parallel_with_resource(resources, func_param_iterator, test_func)
    