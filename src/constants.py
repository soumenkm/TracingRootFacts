import os

MODELS = [
    "mbert",
    "xlmr"
]

WIKI_URI_ROOT = "/raid/speech/soumen/MS_Research/TracingRootFacts/src/wikidata/urls"
MATCH_TYPES = ['full-match', 'partial-match']
PROBE_TYPES = ['acts', 'outs']

DATASET_ROOT = "/raid/speech/soumen/MS_Research/TracingRootFacts/datasets"
CACHE_PATH = os.path.join(DATASET_ROOT, ".cache")
TREx_PATH = os.path.join(DATASET_ROOT, "TREx_multilingual_objects/.cache")
TOKENIZED_DATAFRAME_ROOT = os.path.join(DATASET_ROOT, "mlama_relations")

DATASET_PATH = {"mlama": os.path.join(DATASET_ROOT, "mlama1.1")}
RESULT_ROOT = "/raid/speech/soumen/MS_Research/TracingRootFacts/result"
LOGGING_ROOT = "/raid/speech/soumen/MS_Research/TracingRootFacts/logging"
NEURON_ROOT_ON_DISK = "/raid/speech/soumen/MS_Research/TracingRootFacts/neurons"
PROBER_ROOT_ON_DISK = "/raid/speech/soumen/MS_Research/TracingRootFacts/prober"
HEATMAP_ROOT_ON_DISK = "/raid/speech/soumen/MS_Research/TracingRootFacts/heatmap"
PROBER_ANALYSIS_ROOT_ON_DISK = "/raid/speech/soumen/MS_Research/TracingRootFacts/probing_analysis"
GRAELO_WIKI_CACHE = "/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/graelo_wiki_cache"
WIKI_2018DUMP_CACHE = "/raid/speech/soumen/MS_Research/TracingRootFacts/datasets/2018_dump_wiki_cache"

# PREDICTION_ROOT = {
#     "xlmr": os.path.join(RESULT_ROOT, "prediction-xlmr"),
#     "mbert": os.path.join(RESULT_ROOT, "prediction-mbert"),
# }

# EVALUATION_ROOT = {
#     "xlmr": os.path.join(RESULT_ROOT, "evaluation-xlmr"),
#     "mbert": os.path.join(RESULT_ROOT, "evaluation-mbert"),
# }

# INTERVENTION_ROOT = {
#     "xlmr": os.path.join(RESULT_ROOT, "neuron-intervention-xlmr"),
#     "mbert": os.path.join(RESULT_ROOT, "neuron-intervention-mbert"),
# }

# DATASET_ROOT = {
#     "xlmr": os.path.join(RESULT_ROOT, "dataset-xlmr"),
#     "mbert": os.path.join(RESULT_ROOT, "dataset-mbert"),
# }

# NEURONS_ROOT = {
#     "xlmr": os.path.join(NEURON_ROOT_ON_DISK, "xlmr"),
#     "mbert": os.path.join(NEURON_ROOT_ON_DISK, "mbert"),
# }

# PROBER_ROOT = {
#     "xlmr": os.path.join(PROBER_ROOT_ON_DISK, "xlmr"),
#     "mbert": os.path.join(PROBER_ROOT_ON_DISK, "mbert"),
# }

# PROBER_ANALYSIS_ROOT = {
#     "xlmr": os.path.join(PROBER_ANALYSIS_ROOT_ON_DISK, "xlmr"),
#     "mbert": os.path.join(PROBER_ANALYSIS_ROOT_ON_DISK, "mbert"),
# }

# HEATMAP_ROOT = {
#     "xlmr": os.path.join(HEATMAP_ROOT_ON_DISK, "xlmr"),
#     "mbert": os.path.join(HEATMAP_ROOT_ON_DISK, "mbert"),
# }

# TREx_PATH_ENTITY_ROOT = {
#     "xlmr": os.path.join(TREx_PATH, "entities-xlmr"),
#     "mbert": os.path.join(TREx_PATH, "entities-mbert"),
# }

PREDICTION_ROOT = {model_name: os.path.join(RESULT_ROOT, f"prediction-{model_name}") for model_name in MODELS}
EVALUATION_ROOT = {model_name: os.path.join(RESULT_ROOT, f"evaluation-{model_name}") for model_name in MODELS}
INTERVENTION_ROOT = {model_name: os.path.join(RESULT_ROOT, f"{model_name}") for model_name in MODELS}
DATASET_ROOT = {model_name: os.path.join(RESULT_ROOT, f"dataset-{model_name}") for model_name in MODELS}
NEURONS_ROOT = {model_name: os.path.join(NEURON_ROOT_ON_DISK, model_name) for model_name in MODELS}
PROBER_ROOT = {model_name: os.path.join(PROBER_ROOT_ON_DISK, model_name) for model_name in MODELS}
PROBER_ANALYSIS_ROOT = {model_name: os.path.join(PROBER_ANALYSIS_ROOT_ON_DISK, model_name) for model_name in MODELS}
HEATMAP_ROOT = {model_name: os.path.join(HEATMAP_ROOT_ON_DISK, model_name) for model_name in MODELS}
TREx_PATH_ENTITY_ROOT = {model_name: os.path.join(TREx_PATH, f"entities-{model_name}") for model_name in MODELS}

os.makedirs(TOKENIZED_DATAFRAME_ROOT, exist_ok=True)
# os.makedirs(GRAELO_WIKI_CACHE, exist_ok=True)
# os.makedirs(WIKI_2018DUMP_CACHE, exist_ok=True)

for model in MODELS:
    # os.makedirs(PREDICTION_ROOT[model], exist_ok=True)
    # os.makedirs(EVALUATION_ROOT[model], exist_ok=True)
    # os.makedirs(INTERVENTION_ROOT[model], exist_ok=True)
    os.makedirs(DATASET_ROOT[model], exist_ok=True)
    # os.makedirs(NEURONS_ROOT[model], exist_ok=True)
    # os.makedirs(PROBER_ROOT[model], exist_ok=True)
    # os.makedirs(PROBER_ANALYSIS_ROOT[model], exist_ok=True)
    # os.makedirs(HEATMAP_ROOT[model], exist_ok=True)
    # os.makedirs(TREx_PATH_ENTITY_ROOT[model], exist_ok=True)

LOADING_FILES_LAMBDA = {
    "evaluate_obj_distribution": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "evaluate_obj_distribution.pkl"
    ),
    "get_gold_matrix_per_obj":  lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "gold_matrix_per_lang_obj.pkl"
    ),
    "get_gold_matrix_per_uuid":  lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "gold_matrix_per_lang_uuid.pkl"
    ),
    "get_full_match_matrix_by_obj": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "full_match_matrix_per_lang_obj.pkl"
    ),
    "get_full_match_matrix_by_uuid": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "full_match_matrix_per_lang_uuid.pkl"
    ),
    "get_exact_match_matrix_by_uuid": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "exact_match_matrix_per_lang_uuid.pkl"
    ),
    "get_partial_match_matrix_by_uuid": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "partial_match_matrix_per_lang_uuid.pkl"
    ),
    "p1_evaluate_parallel": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "p1_scores.pkl"
    ),
    "single_multi_evaluate_parallel": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "single_multi-tokens_p1_scores.pkl"
    ),
    "get_all_and_matched_uuid_lsts": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "get_all_and_matched_uuid_lsts.pkl"
    ),
    "calculate_langsim_by_objectwise_p1_with_rel": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "language_similarity_by_objectwise_p1.pkl"
    ),
    "calculate_langsim_by_objectwise_p1_without_rel": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "language_similarity_by_objectwise_p1_without_rel.pkl"
    ),
    "get_wiki_matches_matrix_from_huggingface": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "wiki_and_probing_matching_matrix_from_huggingface.pkl"
    ),
    "get_wiki_matches_matrix_from_dumped_wiki_abstract": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "wiki_and_probing_matching_matrix_from_dumped_wiki_data.pkl"
    ),
    "get_wiki_matches_matrix_from_dumped_wiki_title": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "wiki_and_probing_matching_matrix_from_dumped_wiki_title.pkl"
    ),
    "get_wiki_matches_matrix_from_dumped_wiki_article": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "wiki_and_probing_matching_matrix_from_dumped_wiki_article.pkl"
    ),
    "get_wiki_matches_matrix_from_tokenized_wiki_article": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "wiki_and_probing_matching_matrix_from_tokenized_wiki_article.pkl"
    ),
    "_get_title_object_subject_matchings": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "object_subject_matchings_in_title.pkl"
    ),
    "_get_wiki_matches_resource_from_dumped_wiki_abstract": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "matches_resource_from_dumped_wiki_abstract.pkl"
    ),
    "_get_wiki_matches_resource_from_tokenized_wiki_article": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "matches_resource_from_tokenized_wiki_article.pkl"
    ),
    "_get_subject_object_cooccurence_in_abstract": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "subject_object_cooccurence_from_abstract.pkl"
    ),
    "_get_subject_object_cooccurence_in_article": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "subject_object_cooccurence_from_article.pkl"
    ),
    "get_subject_object_cooccurence_in_tokenized_article": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "subject_object_cooccurence_from_tokenized_article.pkl"
    ),
    "_get_correct_wrong_prediction_of_inwiki_fk": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "get_correct_wrong_prediction_counts_of_inwiki_fk.pkl"
    ),
    "get_correct_wrong_prediction_of_in_tokenized_wiki_fk": lambda dataset: os.path.join(
        EVALUATION_ROOT[dataset.model_name], "get_correct_wrong_prediction_counts_of_in_tokenized_article_fk.pkl"
    ),
    "language_distance": os.path.join(RESULT_ROOT, "z_language_distance.pkl"),
    "load_objects": lambda lang, dataset: os.path.join(
        TREx_PATH_ENTITY_ROOT[dataset.model_name],
        f"object-{lang}.pkl",
    ),
    "load_subjects": lambda lang, dataset: os.path.join(
        TREx_PATH_ENTITY_ROOT[dataset.model_name],
        f"subject-{lang}.pkl",
    ),
    "load_retraining_text_by_lang": lambda lang, dataset: os.path.join(
        GRAELO_WIKI_CACHE,
        f"retrainning_corpus_of_language-{lang}.pkl"),
    "probe_objs": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], rel, f"{match_type}-{probing_type}-probe_objs.pkl",
    ),
    "probe_langs": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], rel, f"{match_type}-{probing_type}-probe_langs.pkl",
    ),
    "probe_objs_per_lang": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], rel, f"{match_type}-{probing_type}-probe_objs_per_lang.pkl",
    ),
    "probe_langs_per_obj": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], rel, f"{match_type}-{probing_type}-probe_langs_per_obj.pkl",
    ),
    "probe_uuid_per_rel": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], rel, f"{match_type}-{probing_type}-probe_uuid_per_rel.pkl",
    ),
    "probe_uuids": lambda model_name, match_type, probing_type, rel: os.path.join(
        PROBER_ROOT[model_name], f"{match_type}-{probing_type}-probe_uuids.pkl",
    ),

    "get_obj_info": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_obj_uris.pkl"),
    "get_sub_info": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_sub_uris.pkl"),
    "get_rel_info": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_rel_uris.pkl"),
    "get_uuid_info": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_uuid_info.pkl"),
    "get_uuid_info_plain": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_uuid_plain_info.pkl"),
    "get_uuid_info_all_lang": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_uuid_info_for_all_langs.pkl"),
    "get_uuid_info_per_lang": lambda dataset: os.path.join(CACHE_PATH, f"{dataset.model_name}_uuid_info_per_langs.pkl"),
    "get_lang2objs": lambda dataset: os.path.join(CACHE_PATH, "lang2objs.pkl"),
    "get_lang2subs": lambda dataset: os.path.join(CACHE_PATH, "lang2subs.pkl"),
    "get_rel_obj_pairs": lambda dataset: os.path.join(CACHE_PATH, "rel_obj_pairs.pkl"),
    "get_subs_per_rel": lambda dataset: os.path.join(CACHE_PATH, "subs_per_rel.pkl"),
    "get_objs_per_rel": lambda dataset: os.path.join(CACHE_PATH, "objs_per_rel.pkl"),
}

if __name__ == "__main__":
    print(PROBER_ROOT)