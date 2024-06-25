import ast
import json
import os

import _pickle as pickle
import pandas as pd
from tqdm import tqdm
from langcodes import Language

from constants import (CACHE_PATH, DATASET_PATH, DATASET_ROOT,
                       TOKENIZED_DATAFRAME_ROOT)
from utils import load_objects, load_subjects, loader, strip_space

pd.set_option("mode.chained_assignment", None)

class MaskedDataset:
    def __init__(self, data_type="mlama", model_name="mbert", reload=False):
        file_name = os.path.join(DATASET_ROOT[model_name], f"{data_type}-dataset.pkl")
        if os.path.exists(file_name) and reload == False:
            print(f"Load pre-saved file {file_name} for MaskedDataset instance")
            with open(file_name, "rb") as fp:
                self.__dict__.update(pickle.load(fp))
            return
        print(f"Initalizing dataset of the type - {data_type}")
        self.datapath = DATASET_PATH[data_type]
        self.langs = []
        self.name = data_type
        self.model_name = model_name
        if self.model_name == "xlmr":
            self.tokenized_obj_root = os.path.join(
                TOKENIZED_DATAFRAME_ROOT, "xlm-roberta-large"
            )
            self.mask_token = "<mask>"
        elif self.model_name == "mbert":
            self.tokenized_obj_root = os.path.join(
                TOKENIZED_DATAFRAME_ROOT, "bert-base-multilingual-cased"
            )
            self.mask_token = "[MASK]"
        else:
            raise Exception(f"Undefined model name {model_name}")

        if data_type != "mlama":
            raise Exception("The type of data is not correctly specified")

        data = []
        self.lang2rels = {}
        for lang in tqdm(os.listdir(self.datapath)):
            df = self._read_lama_dataset(lang, False)
            self.lang2rels[lang] = sorted(list(set(df['relid'].tolist())))
            self.langs.append(lang)
            data.append(df)
        
        rel2langs = {}
        for lang in self.langs:
            for rel in self.lang2rels[lang]:
                rel2langs.setdefault(rel, set()).add(lang)
        self.rel2langs = {rel: sorted(list(_langs)) for rel, _langs in rel2langs.items()}

        self.data = pd.concat(data)
        self.rels = sorted(list(set(self.data['relid'].tolist())))
        self.sub_info = self.get_sub_info()
        self.obj_info = self.get_obj_info()
        self.rel_info = self.get_rel_info()
        self.uuid_info = self.get_uuid_info()
        self.uuid_info_plain = self.get_uuid_info_plain()
        self.uuid_info_all_lang = self.get_uuid_info_all_lang()
        with open(file_name, "wb") as fp:
            pickle.dump(self.__dict__, fp)

    def replace_with_mask(self, sentences, mask_num):
        return [
            s.replace("[Y]", " ".join(mask_num * (self.mask_token,))) for s in sentences
        ]

    def get_sorted_langs(self):
        return sorted(self.langs)

    def get_lang(self, lang):
        return self.data[self.data["lang"] == lang]

    def get_lang_type(self, lang, rel, tokenizer=None, reload=False) -> pd.DataFrame:
        if len(self.data[(self.data["lang"]==lang) & (self.data["relid"]==rel)]) == 0:
            raise ValueError(F"{lang} - {rel} is empty")
        dump_file = os.path.join(self.tokenized_obj_root, lang, f"{rel}.csv")
        if os.path.exists(dump_file) and reload is False:
            df = pd.read_csv(dump_file, index_col=0).reset_index(drop=True)
            return df.set_index("id_by_lang")
        else:
            try:
                if tokenizer is None:
                    from transformers import AutoTokenizer, BertTokenizer
                    if self.model_name == 'xlmr':
                        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
                    if self.model_name == 'mbert':
                        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
                df = self._save_tokenize_objs(lang, rel, tokenizer)
                return df[(df["lang"] == lang) & (df["relid"] == rel)]
            except Exception as e:
                raise NotImplementedError(f"Failed to get_lang_type() for {lang}-{rel}") from e

    def lang_rel_iter(self, cand_langs=None, cand_rels=None):
        cand_langs = cand_langs if cand_langs else self.langs
        cand_rels = cand_rels if cand_rels else self.rels
        for lang in cand_langs:
            for rel in self.get_rels_in_lang(lang, cand_rels):
                if rel in cand_rels:
                    yield lang, rel
            
    def get_langs_in_rel(self, rel, cand_langs=None) -> list:
        cand_langs = cand_langs if cand_langs else self.langs
        langs = []
        for lang in self.rel2langs[rel]:
            if lang in cand_langs:
                langs.append(lang)
        return langs
    
    def get_rels_in_lang(self, lang, cand_rels=None):
        cand_rels = cand_rels if cand_rels else self.rels
        rels = []
        for rel in self.lang2rels[lang]:
            if rel in cand_rels:
                rels.append(rel)
        return rels

    @loader
    def get_subs_per_rel(self, reload=False) -> dict:
        rel2sub_uris = {}
        for lang, rel in tqdm(self.lang_rel_iter(), desc="Collecting subjects per rel"):
            df = self.get_lang_type(lang, rel)
            if rel not in rel2sub_uris:
                rel2sub_uris[rel] = set(df.sub_uri)
            else:
                rel2sub_uris[rel].update(set(df.sub_uri))
        for rel in rel2sub_uris:
            rel2sub_uris[rel] = sorted(list(rel2sub_uris[rel]))
        return rel2sub_uris
    
    @loader
    def get_objs_per_rel(self, reload=False) -> dict:
        rel2obj_uris = {}
        for lang, rel in tqdm(self.lang_rel_iter(), desc="Collecting objects per rel"):
            df = self.get_lang_type(lang, rel)
            if rel not in rel2obj_uris:
                rel2obj_uris[rel] = set(df.obj_uri)
            else:
                rel2obj_uris[rel].update(set(df.obj_uri))
        for rel in rel2obj_uris:
            rel2obj_uris[rel] = sorted(list(rel2obj_uris[rel]))
        return rel2obj_uris
    
    def get_objs_in_rel(self, rel: str) -> list:
        return self.get_objs_per_rel()[rel]

    def get_subs_in_rel(self, rel: str) -> list:
        return self.get_subs_per_rel()[rel]

    def get_tokenized_objs_in_rel(self, rel: int) -> dict:
        """
        Returns:
            objs: {
                "en": {
                    1: {'Bengali': [151303], 'Chinese': [76438],...},
                    2: {'Armenian': [135665, 19], 'Belarusian': [113852, 3378], ...},
                    ...
                }, ...
                "fr": {...}
            }
        """
        return {
            lang: load_objects(lang=lang, dataset=self)[rel]
            for lang in self.langs
        }
    
    @loader
    def get_rel_obj_pairs(self, reload=False):
        rel_objs = []
        for rel in self.rels:
            for obj in self.get_objs_in_rel(rel):
                rel_objs.append(f"{rel}-{obj}")
        rel_objs = sorted(rel_objs)
        return rel_objs

    def get_tokenized_objs_per_lang(self, lang: int) -> dict:
        """
        Returns:
            objs: {
                "P1001": {
                    1: {'Bengali': [151303], 'Chinese': [76438],...},
                    2: {'Armenian': [135665, 19], 'Belarusian': [113852, 3378], ...},
                    ...
                }, ...
                "P1002": {...}
            }
        """
        return load_objects(lang=lang, dataset=self)
    
    def get_all_tokenized_objs(self) -> dict:
        """
        Returns:
            objs: {
                'en': {
                    "P1001": {
                        1: {'Bengali': [151303], 'Chinese': [76438],...},
                        2: {'Armenian': [135665, 19], 'Belarusian': [113852, 3378], ...},
                        ...
                    }, ...
                    "P1002": {...}
                }
                
            }
        """
        return {
            lang: load_objects(lang=lang, dataset=self)
            for lang in self.langs
        }
    
    def get_tokenized_subs_in_rel(self, rel):
        """
        Returns:
            subjs: similar to get_objs_in_rel
        """
        return {
            lang: load_subjects(lang=lang, dataset=self)[rel]
            for lang in self.langs
        }

    def get_tokenized_subs_per_lang(self, lang):
        """
        Returns:
            subjs: similar to get_objs_in_rel
        """
        return load_subjects(lang=lang, dataset=self)

    def get_line(self, relid, lineid):
        return self.data[
            (self.data["relid"] == relid) & (self.data["lineid"] == lineid)
        ]
    
    def display_obj(self, obj_uri):
        return self.obj_info[obj_uri]['en']['obj']
    
    def display_sub(self, sub_uri):
        return self.sub_info[sub_uri]['en']['sub']
    
    def display_rel(self, rel_uri):
        return self.rel_info[rel_uri]['en']
    
    def display_uuid(self, uuid, lang='en'):
        if lang not in self.uuid_info_all_lang[uuid]:
            print(f"{lang} doesn't has information about {uuid}, display English instead")
            lang = 'en'
        uuid_info = self.uuid_info_all_lang[uuid][lang]
        output = uuid_info['rel']
        output = output.replace('[Y]', uuid_info['obj'])
        output = output.replace('[X]', uuid_info['sub'])
        return output
    
    @loader
    def get_uuid_info(self, reload=False):
        """
        Returns:
            rel2uuid: {
                "P103": {
                    '7d58f005-5166-4af7-a2e0-c960de153441': {
                        'sub_uri': 'Q7251',
                        'sub': 'Alan Turing',
                        'obj_uri': 'Q8078',
                        'obj': 'logic'}, ...
                }, ...
            }
        """
        rel2uuid = {}
        for rel in tqdm(self.get_rels_in_lang('en'), desc="Initalizing uuid info data by English"):
            df = self.get_lang_type('en', rel)
            rel2uuid[rel] = {
                row["uuid"]: {
                    'sub_uri': row["sub_uri"],
                    'sub': row["sub"],
                    'obj_uri': row["obj_uri"],
                    'obj': row["obj"],
                    'langs': ['en']
                }
                for index, row in df.iterrows()
            }
        
        cnt = 0
        for lang, rel in tqdm(self.lang_rel_iter(), desc="Supplementing information from non-English languages"):
            if lang == 'en':
                continue
            df = self.get_lang_type(lang, rel)
            rel2uuid.setdefault(rel, {})
            for index, row in df.iterrows():
                if row['uuid'] not in rel2uuid[rel]:
                    cnt += 1
                    rel2uuid[rel][row["uuid"]] = {
                        'sub_uri': row["sub_uri"],
                        'sub': row["sub"],
                        'obj_uri': row["obj_uri"],
                        'obj': row["obj"],
                        'langs': []
                    }
                rel2uuid[rel][row['uuid']]['langs'].append(lang)
        print(f"{cnt} new uuid information is added from non-English languages")
        return rel2uuid

    @loader
    def get_uuid_info_plain(self, reload=False):
        """
        Returns:{
            '7d58f005-5166-4af7-a2e0-c960de153441': {
                "rel": "[X] is located at [Y]",
                "rel_uri": "P103",
                'sub': 'Alan Turing',
                'sub_uri': 'Q7251',
                'obj': 'logic'
                'obj_uri': 'Q8078',
                'langs': = ['en', 'ja', 'zh', ...]
            }, ...}
        """
        uuid_info_plain = {}
        for rel_uri, uuids_per_rel in tqdm(self.get_uuid_info().items(), desc="Collecting uuid information"):
            for uuid in uuids_per_rel:
                uuid_info_plain[uuid] = uuids_per_rel[uuid]
                uuid_info_plain[uuid].update({
                    "rel_uri": rel_uri,
                    "rel": self.display_rel(rel_uri)})
        return uuid_info_plain

    @loader
    def get_uuid_info_all_lang(self, reload=False):
        """
        Returns:{
            '7d58f005-5166-4af7-a2e0-c960de153441': {
                'en':{
                    "rel": "[X] is located at [Y]",
                    "rel_uri": "P103",
                    'sub': 'Alan Turing',
                    'sub_uri': 'Q7251',
                    'obj': 'logic'
                    'obj_uri': 'Q8078',
                }, ...}
                
        """
        uuid_info = {}
        raw_uuid_info = self.get_uuid_info_plain()
        obj_info = self.get_obj_info()
        sub_info = self.get_sub_info()
        rel_info = self.get_rel_info()
        for uuid in tqdm(list(raw_uuid_info.keys()), desc="Collecting uuid information by languages"):
            uuid_info[uuid] = {}
            for lang in raw_uuid_info[uuid]['langs']:
                uuid_info[uuid][lang] = {
                    "obj_uri": raw_uuid_info[uuid]['obj_uri'],
                    "sub_uri": raw_uuid_info[uuid]['sub_uri'],
                    "rel_uri": raw_uuid_info[uuid]['rel_uri'],
                    "obj": obj_info[raw_uuid_info[uuid]['obj_uri']][lang]['obj'],
                    "sub": sub_info[raw_uuid_info[uuid]['sub_uri']][lang]['sub'],
                    "rel": rel_info[raw_uuid_info[uuid]['rel_uri']][lang],
                }
        return uuid_info

    @loader
    def get_uuid_info_per_lang(self, reload=True):
        """
        Returns:{
            'en': {
                '7d58f005-5166-4af7-a2e0-c960de153441': {
                    "rel": "[X] is located at [Y]",
                    "rel_uri": "P103",
                    'sub': 'Alan Turing',
                    'sub_uri': 'Q7251',
                    'obj': 'logic'
                    'obj_uri': 'Q8078',
                }, ...}}
        """
        uuid_info = self.get_uuid_info_all_lang()
        lang2uuids = {}
        for uuid in tqdm(uuid_info.keys(), desc="Processing uuid information per language"):
            for lang in self.langs:
                if lang in uuid_info[uuid].keys():
                    uuid_info[uuid][lang].update({"uuid": uuid})
                    lang2uuids.setdefault(lang, []).append(uuid_info[uuid][lang])
        return lang2uuids

    @loader
    def get_obj_info(self, reload=False):
        """
        Returns:
            obj_uris: {
                "Q100": {
                    'ms': {'obj': 'Boston', 'obj_ids': [62704], 'obj_tokens': ['▁Boston']},
                    {'obj': '波士顿', 'obj_ids': [7420, 8779, 34960], 'obj_tokens': ['▁', '波', '士', '顿']},
                    ...
                }
            }
        """
        obj_uris = {}
        for lang, rel in tqdm(list(self.lang_rel_iter()), "Loading object information"):
            gold = self.get_lang_type(lang, rel)
            for idx in range(gold.shape[0]):
                obj = gold.iloc[idx,:]["obj"]
                obj_uri = gold.iloc[idx,:]["obj_uri"]
                
                obj_ids = gold.iloc[idx,:]["obj_ids"]
                if isinstance(obj_ids, str):
                    obj_ids = ast.literal_eval(obj_ids)
                obj_ids = strip_space(obj_ids, is_wrapped=False)
                
                obj_tokens = gold.iloc[idx,:]["obj_tokens"]
                if isinstance(obj_tokens, str):
                    obj_tokens = ast.literal_eval(obj_tokens)
                
                if obj_uri not in obj_uris:
                    obj_uris[obj_uri] = {lang: {"obj": obj, "obj_ids": obj_ids, "obj_tokens": obj_tokens}}
                elif lang not in [obj_uris[obj_uri]]:
                    obj_uris[obj_uri].update({lang: {"obj": obj, "obj_ids": obj_ids, "obj_tokens": obj_tokens}})
        return obj_uris

    @loader
    def get_sub_info(self, reload=False):
        """
        Returns:
            obj_uris: {
                "Q100": {
                    'ms': {'obj': 'Boston', 'obj_ids': [62704], 'obj_tokens': ['▁Boston']},
                    {'obj': '波士顿', 'obj_ids': [7420, 8779, 34960], 'obj_tokens': ['▁', '波', '士', '顿']},
                    ...
                }
            }
        """
        sub_uris = {}
        for lang, rel in tqdm(list(self.lang_rel_iter()), "Loading subject information"):
            gold = self.get_lang_type(lang, rel)
            for idx in range(gold.shape[0]):
                sub = gold.iloc[idx, :]["sub"]
                sub_uri = gold.iloc[idx, :]["sub_uri"]
                sub_ids = gold.iloc[idx, :]["sub_ids"]
                if isinstance(sub_ids, str):
                    sub_ids = ast.literal_eval(sub_ids)
                sub_ids = strip_space(sub_ids, is_wrapped=False)
                
                sub_tokens = gold.iloc[idx]["sub_tokens"]
                if isinstance(sub_tokens, str):
                    sub_tokens = ast.literal_eval(sub_tokens)
                
                if sub_uri not in sub_uris:
                    sub_uris[sub_uri] = {lang: {"sub": sub, "sub_ids": sub_ids, "sub_tokens": sub_tokens}}
                elif lang not in [sub_uris[sub_uri]]:
                    sub_uris[sub_uri].update({lang: {"sub": sub, "sub_ids": sub_ids, "sub_tokens": sub_tokens}})
                
                # json.dump(sub_uris, open("temp.json", "w"))
                
        return sub_uris
    
    @loader
    def get_rel_info(self, reload=False):
        rel_infos = {}
        for lang in self.langs:
            root = os.path.join(self.datapath, lang)
            template_path = os.path.join(root, "templates.jsonl")
            with open(template_path, "r") as fp:
                templates = [json.loads(l) for l in fp.readlines()]
                for rel_info in templates:
                    if rel_info['relation'] in rel_infos:
                        rel_infos[rel_info['relation']].update({lang: rel_info['template']})
                    else:
                        rel_infos[rel_info['relation']] = {lang: rel_info['template']}
        return rel_infos
    
    @loader
    def get_lang2objs(self, reload=False):
        """
        Returns:
        - lang2objs
            dict: {
                "en": "Tokyo", "Japan", "China", ..., 
                "ja": "日本", ...
            },
        - lang2objs_uri
            dict: {
                "en": "Q100", "Q101", "Q102", ..., 
                "ja": "Q100", ...
            },
        """
        obj_info = self.get_obj_info()
        lang2objs = {}
        lang2objs_uri = {}
        
        for lang in tqdm(self.langs, desc="Loading object information per language"):
            lang2objs[lang] = []
            lang2objs_uri[lang] = []
            for obj_uri, lang2obj_info in obj_info.items():
                if lang in lang2obj_info:
                    lang2objs[lang].append(str(lang2obj_info[lang]["obj"]))
                    lang2objs_uri[lang].append(obj_uri)
        return lang2objs, lang2objs_uri
    
    @loader
    def get_lang2subs(self, reload=False):
        """
        Returns:
        - lang2subs
            dict: {
                "en": "Tokyo", "Japan", "China", ..., 
                "ja": "日本", ...
            },
        - lang2subs_uri
            dict: {
                "en": "Q100", "Q101", "Q102", ..., 
                "ja": "Q100", ...
            },
        """
        sub_info = self.get_sub_info()
        lang2subs = {}
        lang2subs_uri = {}
        
        for lang in tqdm(self.langs, desc="Loading subject information per language"):
            lang2subs[lang] = []
            lang2subs_uri[lang] = []
            for sub_uri, lang2sub_info in sub_info.items():
                if lang in lang2sub_info:
                    lang2subs[lang].append(str(lang2sub_info[lang]["sub"]))
                    lang2subs_uri[lang].append(sub_uri)
        return lang2subs, lang2subs_uri
    
    def obj2uri(self, name):
        return set(self.data[self.data.obj == name]['obj_uri'].tolist())

    def sub2uri(self, name):
        return set(self.data[self.data.obj == name]['sub_uri'].tolist())

    def display_lang(self, lang, prefix=True):
        if prefix:
            return f"{lang}-{Language.get(lang).display_name()}"
        else:
            return f"{Language.get(lang).display_name()}"

    def _save_tokenize_objs(self, lang, rel, tokenizer):
        """Enginee for `get_lang_type()`"""
        os.makedirs(os.path.join(self.tokenized_obj_root, lang), exist_ok=True)
        df = self.data[(self.data["lang"] == lang) & (self.data["relid"] == rel)]
        df.loc[:, "obj_tokens"] = df["obj"].apply(lambda x: tokenizer.tokenize(x))
        df.loc[:, "obj_ids"] = df["obj_tokens"].apply(
            lambda x: tokenizer.convert_tokens_to_ids(x)
        )
        df.loc[:, "sub_tokens"] = df["sub"].apply(lambda x: tokenizer.tokenize(x))
        df.loc[:, "sub_ids"] = df["sub_tokens"].apply(
            lambda x: tokenizer.convert_tokens_to_ids(x)
        )
        df["id_by_lang"] = df.index.tolist()
        df.to_csv(os.path.join(self.tokenized_obj_root, lang, f"{rel}.csv"), index=False)
        return df

    def _reload_all(self):
        # self._reload_lama_dataset()
        # self._reload_lang_type_dataframe()
        # self.get_uuid_info(reload=True)
        # self.get_uuid_info_plain(reload=True)
        # self.get_obj_info(reload=True)
        # self.get_sub_info(reload=True)
        # self.get_rel_info(reload=True)
        self.get_lang2objs(reload=True)
        self.get_lang2subs(reload=True)
        # self.get_uuid_info_all_lang(reload=True)
        self.get_uuid_info_per_lang(reload=True)
        self.get_subs_per_rel(reload=True)
        self.get_objs_per_rel(reload=True)
        self.get_rel_obj_pairs(reload=True)


        self.__init__(data_type=self.name, model_name=self.model_name, reload=True)

    def _reload_lang_type_dataframe(self, cand_langs=None, cand_rels=None):
        """Wrapper of `get_lang_type()`, to reload"""
        from transformers import AutoTokenizer, BertTokenizer
        cand_langs = cand_langs if cand_langs != None else self.langs 
        cand_rels = cand_rels if cand_rels != None else self.rels 
        if self.model_name == 'xlmr':
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
            for lang, rel in tqdm(list(self.lang_rel_iter(cand_langs=cand_langs, cand_rels=cand_rels))):
                self.get_lang_type(lang, rel, tokenizer=tokenizer, reload=True)        
        if self.model_name == 'mbert':
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            for lang, rel in tqdm(list(self.lang_rel_iter(cand_langs=cand_langs, cand_rels=cand_rels))):
                self.get_lang_type(lang, rel, tokenizer=tokenizer, reload=True)        
    
    def _reload_lama_dataset(self):
        for lang in tqdm(self.langs, desc="Reloading lama dataset"):
            self._read_lama_dataset(lang, reload=True)

    def _read_lama_dataset(self, lang, reload=False):
        dumppath = os.path.join(CACHE_PATH, lang + "_mlama.feather")
        if os.path.exists(dumppath) and reload is False:
            return pd.read_feather(dumppath)
        dataset = pd.DataFrame()

        root = os.path.join(self.datapath, lang)
        template_path = os.path.join(root, "templates.jsonl")
        with open(template_path, "r") as fp:
            templates = [json.loads(l) for l in fp.readlines()]
        for rel_info in templates:
            rel_id = rel_info["relation"]
            temp = rel_info["template"]
            relpath = os.path.join(root, rel_id + ".jsonl")
            if os.path.exists(relpath) is False or os.stat(relpath).st_size == 0:
                continue
            with open(relpath, "r") as fp:
                for rel in [json.loads(l) for l in fp.readlines()]:
                    item = {
                        "sent": [temp.replace("[X]", rel["sub_label"])],
                        "lang": [lang],
                        "relid": [rel_id],
                        "uuid": [rel["uuid"]],
                        "lineid": [str(rel["lineid"])],
                        "obj": [rel["obj_label"]],
                        "obj_uri": [rel["obj_uri"]],
                        "sub": [rel["sub_label"]],
                        "sub_uri": [rel["sub_uri"]],
                    }
                    dataset = pd.concat([dataset, pd.DataFrame(item)])
        dataset.reset_index().to_feather(dumppath)
        return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer, BertTokenizer


    from utils import load_objects, load_subjects
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    dataset = MaskedDataset(model_name="xlmr", reload=False)
    # for lang in dataset.langs:
    #     load_objects(lang=lang, dataset=dataset, tokenizer=tokenizer, reload=True)
    #     load_subjects(lang=lang, dataset=dataset, tokenizer=tokenizer, reload=True)
    dataset._reload_all()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    # dataset = MaskedDataset(model_name="xlmr", reload=True)

    # from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
    #                                 as_completed)

    # with ThreadPoolExecutor(50) as executor:
    #     lang_rel = []
    #     for lang in mlama.langs:
    #         objs = load_objects(lang=lang, dataset=mlama)
    #         for rel in objs.keys():
    #             lang_rel.append([lang, rel])
    #     futures = [
    #         executor.submit(mlama._save_tokenize_objs, lang, rel, tokenizer)
    #         for lang, rel in lang_rel
    #     ]
    #     for future in as_completed(futures):
    #         future.result()
