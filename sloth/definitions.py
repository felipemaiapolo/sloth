import numpy as np

###
benchs_names_list = [['MMLU','ARC','HellaSwag','Winogrande','TruthfulQA','GSM8K'],
                     ['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO'],
                     ['MMLU','ARC','HellaSwag','Winogrande','TruthfulQA','GSM8K','IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']] 

###
methods_names = ['logF_sigmoid', 'logF_trainlink', 'logF_sigmoid_faminter', 'logF_trainlink_faminter',
                 'logSlogT_sigmoid', 'logSlogT_trainlink', 'logSlogT_sigmoid_faminter', 'logSlogT_trainlink_faminter',
                 'pca_d=1', 'pca_d=2', 'pca_d=3', 'pca_d=4',
                  'factor_sigmoid_d=1','factor_sigmoid_d=2','factor_sigmoid_d=3','factor_sigmoid_d=4',
                  'factor_trainlink_d=1','factor_trainlink_d=2','factor_trainlink_d=3','factor_trainlink_d=4',
                  'factor_sigmoid_faminter_d=1','factor_sigmoid_faminter_d=2','factor_sigmoid_faminter_d=3','factor_sigmoid_faminter_d=4',
                  'factor_trainlink_faminter_d=1','factor_trainlink_faminter_d=2','factor_trainlink_faminter_d=3','factor_trainlink_faminter_d=4']

### Test families based on the availability in the dataset for different benchmarks
test_families = {1:[[['bloom'],
                           ['codegen-nl'],
                           ['codellama'],
                           ['deepseek-coder-base'],
                           ['pythia','dolly-v2'],
                           ['falcon'],
                           ['gemma', 'gemma-it','sauerkrautlm-gemma'],
                           ['gpt-j-neo-neox'], 
                           ['internlm2'],
                           ['meta-llama-3', 'meta-llama-3-instruct'],
                           ['mpt', 'mpt-chat','mpt-instruct'],
                           ['olmo'],
                           ['opt'],
                           ['qwen2'],
                           ['rwkv-4-pile'],
                           ['starcoder2'],
                           ['stablelm-base-alpha'],
                           ['xglm'],
                           ['yi-1.5', 'yi-1.5-chat','dolphin-2.9.1-yi-1.5']],
                          [['bloom'],
                           ['pythia','dolly-v2'],
                           ['falcon','falcon-instruct'],
                           ['gemma-2', 'gemma-2-it'],
                           ['gpt-j-neo-neox'], 
                           ['meta-llama-3', 'meta-llama-3-instruct','llama-3-sauerkrautlm-instruct'],
                           ['olmo'],
                           ['opt'],
                           ['qwen2','qwen2-instruct','dolphin-2.9.2-qwen2'],
                           ['starcoder2'],
                           ['smollm', 'smollm-instruct'],
                           ['yi-1.5', 'yi-1.5-chat','dolphin-2.9.1-yi-1.5']],
                          [['bloom'],
                           ['pythia','dolly-v2'],
                           ['falcon'],
                           ['gemma', 'gemma-it', 'sauerkrautlm-gemma'],
                           ['gpt-j-neo-neox'], 
                           ['meta-llama-3', 'meta-llama-3-instruct'],
                           ['olmo'],
                           ['opt'],
                           ['qwen2'],
                           ['starcoder2'],
                           ['yi-1.5', 'yi-1.5-chat','dolphin-2.9.1-yi-1.5']]],
                 2:[[['bloom'],
                             ['codellama'],
                             ['deepseek-coder-base'], 
                             ['falcon'],
                             ['gpt-j-neo-neox'], 
                             ['llama-2', 'llama-2-chat'],
                             ['open_llama_'], 
                             ['opt'], 
                             ['pythia'], 
                             ['qwen1.5', 'qwen1.5-chat'],
                             ['qwen2'],
                             ['rwkv-4-pile'],
                             ['starcoder2'],
                             ['xglm'],
                             ['yi-1.5', 'yi-1.5-chat']],
                          [['bloom'],
                             ['llama-2', 'llama-2-chat'],
                             ['orca_mini_v3_'],
                             ['pythia','dolly-v2'],
                             ['qwen1.5', 'qwen1.5-chat'],
                             ['qwen2','qwen2-instruct'],
                             ['smollm', 'smollm-instruct'], 
                             ['starcoder2'],
                             ['yi-1.5', 'yi-1.5-chat']],
                          [['bloom'],
                             ['llama-2', 'llama-2-chat'], 
                             ['pythia'], 
                             ['qwen1.5','qwen1.5-chat'],
                             ['qwen2'], 
                             ['starcoder2'], 
                             ['yi-1.5','yi-1.5-chat']]]}
    
### Lower bounds for benchmarks (computed using 1/#choices)
# for Open LLM LB 2, check https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
BBH_ns ={'bbh_boolean_expressions':250,'bbh_causal_judgement':187,'bbh_date_understanding':250,'bbh_disambiguation_qa':250,
         'bbh_formal_fallacies':250,'bbh_geometric_shapes':250,'bbh_hyperbaton':250,'bbh_logical_deduction_five_objects':250,
         'bbh_logical_deduction_seven_objects':250,'bbh_logical_deduction_three_objects':250,'bbh_movie_recommendation':250,
         'bbh_navigate':250,'bbh_object_counting':250,'bbh_penguins_in_a_table':146,'bbh_reasoning_about_colored_objects':250,
         'bbh_ruin_names':250,'bbh_salient_translation_error_detection':250,'bbh_snarks':178,'bbh_sports_understanding':250,
         'bbh_temporal_sequences':250,'bbh_tracking_shuffled_objects_five_objects':250,'bbh_tracking_shuffled_objects_seven_objects':250,
         'bbh_tracking_shuffled_objects_three_objects':250,'bbh_web_of_lies':250}
BBH_cs ={'bbh_boolean_expressions':1/2,'bbh_causal_judgement':1/2,'bbh_date_understanding':1/6,'bbh_disambiguation_qa':1/3,
         'bbh_formal_fallacies':1/2,'bbh_geometric_shapes':1/11,'bbh_hyperbaton':1/2,'bbh_logical_deduction_five_objects':1/5,
         'bbh_logical_deduction_seven_objects':1/7,'bbh_logical_deduction_three_objects':1/3,'bbh_movie_recommendation':1/6,
         'bbh_navigate':1/2,'bbh_object_counting':1/19,'bbh_penguins_in_a_table':1/5,'bbh_reasoning_about_colored_objects':1/18,
         'bbh_ruin_names':1/6,'bbh_salient_translation_error_detection':1/6,'bbh_snarks':1/2,'bbh_sports_understanding':1/2,
         'bbh_temporal_sequences':1/4,'bbh_tracking_shuffled_objects_five_objects':1/5,'bbh_tracking_shuffled_objects_seven_objects':1/7,
         'bbh_tracking_shuffled_objects_three_objects':1/3,'bbh_web_of_lies':1/2}

MUSR_ns ={'musr_murder_mysteries':250,
          'musr_object_placements':256,
          'musr_team_allocation':250}
MUSR_cs ={'musr_murder_mysteries':1/2,
          'musr_object_placements':1/5,
          'musr_team_allocation':1/3}

lower_bounds = {'MMLU':.25,
                 'HellaSwag':.25,
                 'Winogrande':.5,
                 'GSM8K':0,
                 'ARC':.25,
                 'TruthfulQA':.31, #this number is computed by loading the leaderboard data and computing the 1st percentile of scores
                 'IFEval':0,
                 'MATH Lvl 5':0,
                 'MMLU-PRO':.1,
                 'BBH':np.sum([BBH_ns[k]*BBH_cs[k] for k in BBH_cs.keys()])/np.sum([BBH_ns[k] for k in BBH_cs.keys()]),
                 'GPQA':.25,
                 'MUSR':np.sum([MUSR_ns[k]*MUSR_cs[k] for k in MUSR_cs.keys()])/np.sum([MUSR_ns[k] for k in MUSR_cs.keys()])}