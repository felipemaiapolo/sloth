import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

def prepare_data(chosen_scenarios, scenarios, data):
    
    """
    Prepare the data by determining the positions of items within each scenario and subscenario.
    
    Parameters:
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.
    
    Returns:
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    """
    
    i = 0
    subscenarios_position = {}
    
    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in chosen_scenarios:
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data['data'][sub]['correctness'].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1
    
    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in chosen_scenarios:
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position
    
def download_model_correctness(models):

    subscenarios = []
    for sub in list(lb_scenarios.values()):
        subscenarios+=sub

    data = {}
    for model in tqdm(models):
        data[model] = {}
        for s in subscenarios:
            data[model][s] = {}
            data[model][s]['correctness'] = None
            data[model][s]['dates'] = None
            
    skipped = 0
    log = []
    for model in tqdm(models):
        skipped_aux=0
        data = {}
        for s in subscenarios:
            if 'arc' in s: metric = 'acc_norm'
            elif 'hellaswag' in s: metric = 'acc_norm'
            elif 'truthfulqa' in s: metric = 'mc2'
            else: metric = 'acc'
    
            data[s] = load_dataset(model, s)
             
    return data
    
def filter(s):
    try:s = s.split("/")[1]
    except: s = s
    try:s = s.split("__")[1]
    except: s = s
    return s.lower().replace("-hf","").replace("_","").replace("-","")
    
def search(s, s_list):
    scores = [fuzz.token_sort_ratio(filter(s), filter(s_try)) for s_try in s_list]
    return [s_list[np.argmax(scores)], np.max(scores)]

def create_irt_data(correctness, models_part, models_full):
    Y = 1*(correctness>.5)
    M = np.zeros((len(models_part), len(models_full)))
    for i,mod in enumerate(models_part):
        M[i,np.argmax(np.array(models_full)==mod)]=1
    return Y, M

def create_arena_data(arena_data, models_full):
    Y_arena = np.array(arena_data.loc[:,['winner_model_a', 'winner_model_b', 'winner_tie']])
    M_arena = np.zeros((Y_arena.shape[0], len(models_full)))
    for i in tqdm(range(Y_arena.shape[0])):
        M_arena[i, np.argmax(np.array(models_full) == filter(arena_data.model_a[i]))] = 1
        M_arena[i, np.argmax(np.array(models_full) == filter(arena_data.model_b[i]))] = -1
    return Y_arena, M_arena
    
def sigmoid(z):
    
    """
    Compute the sigmoid function for the input z.
    
    Parameters:
    - z: A numeric value or numpy array.
    
    Returns:
    - The sigmoid of z.
    """
    
    return 1/(1+np.exp(-z))

def item_curve(theta, a, b):
    
    """
    Compute the item response curve for given parameters.
    
    Parameters:
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.
    
    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)
    
def prepare_data(scenarios, data):
    
    """
    Prepare the data by determining the positions of items within each scenario and subscenario.
    
    Parameters:
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.
    
    Returns:
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    """
    
    i = 0
    subscenarios_position = {}
    
    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in scenarios.keys():
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data['data'][sub]['correctness'].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1
    
    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in scenarios.keys():
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position

def create_responses(scenarios, data):
    
    """
    Create a matrix of responses for the chosen scenarios.
    
    Parameters:
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.
    
    Returns:
    - A numpy array of responses for the chosen scenarios.
    """

    responses = [np.vstack([data['data'][sub]['correctness'] for sub in scenarios[scenario]]).T for scenario in scenarios.keys()]
    responses = np.hstack(responses)
    return responses

alpaca_scenarios = {'alpaca_v2':['alpaca_v2']}
lb_scenarios = {'harness_truthfulqa_mc_0':['harness_truthfulqa_mc_0'],
                 'gsm8k':['harness_gsm8k_5'], 
                 'winogrande':['harness_winogrande_5'], 
                 'arc':['harness_arc_challenge_25'], 
                 'hellaswag':['harness_hellaswag_10'],
                 'mmlu':['harness_hendrycksTest_abstract_algebra_5', 
                         'harness_hendrycksTest_anatomy_5', 
                         'harness_hendrycksTest_astronomy_5', 
                         'harness_hendrycksTest_business_ethics_5', 
                         'harness_hendrycksTest_clinical_knowledge_5', 
                         'harness_hendrycksTest_college_biology_5', 
                         'harness_hendrycksTest_college_chemistry_5', 
                         'harness_hendrycksTest_college_computer_science_5', 
                         'harness_hendrycksTest_college_mathematics_5', 
                         'harness_hendrycksTest_college_medicine_5', 
                         'harness_hendrycksTest_college_physics_5', 
                         'harness_hendrycksTest_computer_security_5', 
                         'harness_hendrycksTest_conceptual_physics_5', 
                         'harness_hendrycksTest_econometrics_5', 
                         'harness_hendrycksTest_electrical_engineering_5', 
                         'harness_hendrycksTest_elementary_mathematics_5', 
                         'harness_hendrycksTest_formal_logic_5', 
                         'harness_hendrycksTest_global_facts_5', 
                         'harness_hendrycksTest_high_school_biology_5', 
                         'harness_hendrycksTest_high_school_chemistry_5', 
                         'harness_hendrycksTest_high_school_computer_science_5', 
                         'harness_hendrycksTest_high_school_european_history_5', 
                         'harness_hendrycksTest_high_school_geography_5', 
                         'harness_hendrycksTest_high_school_government_and_politics_5', 
                         'harness_hendrycksTest_high_school_macroeconomics_5', 
                         'harness_hendrycksTest_high_school_mathematics_5', 
                         'harness_hendrycksTest_high_school_microeconomics_5', 
                         'harness_hendrycksTest_high_school_physics_5', 
                         'harness_hendrycksTest_high_school_psychology_5', 
                         'harness_hendrycksTest_high_school_statistics_5', 
                         'harness_hendrycksTest_high_school_us_history_5', 
                         'harness_hendrycksTest_high_school_world_history_5', 
                         'harness_hendrycksTest_human_aging_5', 
                         'harness_hendrycksTest_human_sexuality_5', 
                         'harness_hendrycksTest_international_law_5', 
                         'harness_hendrycksTest_jurisprudence_5', 
                         'harness_hendrycksTest_logical_fallacies_5', 
                         'harness_hendrycksTest_machine_learning_5', 
                         'harness_hendrycksTest_management_5', 
                         'harness_hendrycksTest_marketing_5', 
                         'harness_hendrycksTest_medical_genetics_5', 
                         'harness_hendrycksTest_miscellaneous_5', 
                         'harness_hendrycksTest_moral_disputes_5', 
                         'harness_hendrycksTest_moral_scenarios_5', 
                         'harness_hendrycksTest_nutrition_5', 
                         'harness_hendrycksTest_philosophy_5', 
                         'harness_hendrycksTest_prehistory_5', 
                         'harness_hendrycksTest_professional_accounting_5', 
                         'harness_hendrycksTest_professional_law_5', 
                         'harness_hendrycksTest_professional_medicine_5', 
                         'harness_hendrycksTest_professional_psychology_5',
                         'harness_hendrycksTest_public_relations_5', 
                         'harness_hendrycksTest_security_studies_5', 
                         'harness_hendrycksTest_sociology_5', 
                         'harness_hendrycksTest_us_foreign_policy_5', 
                         'harness_hendrycksTest_virology_5', 
                         'harness_hendrycksTest_world_religions_5']}
helm_lite_scenarios = {'commonsense:dataset=openbookqa,method=multiple_choice_joint,':['commonsense:dataset=openbookqa,method=multiple_choice_joint,'],
                       'gsm:':['gsm:'],
                       'med_qa:':['med_qa:'],
                       'legalbench':[#'legalbench:subset=abercrombie,',
                                     #'legalbench:subset=corporate_lobbying,',
                                     'legalbench:subset=function_of_decision_section,',
                                     'legalbench:subset=proa,',
                                     'legalbench:subset=international_citizenship_questions,'],
                      'math':['math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
                              'math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,',],
                      'mmlu':['mmlu:subject=abstract_algebra,method=multiple_choice_joint,',
                              'mmlu:subject=college_chemistry,method=multiple_choice_joint,',
                              'mmlu:subject=computer_security,method=multiple_choice_joint,',
                              'mmlu:subject=econometrics,method=multiple_choice_joint,',
                              'mmlu:subject=us_foreign_policy,method=multiple_choice_joint,'],
                      'narrative_qa:':['narrative_qa:'],
                      'natural_qa:mode=closedbook,':['natural_qa:mode=closedbook,'],
                      'natural_qa:mode=openbook_longans,':['natural_qa:mode=openbook_longans,'],
                      'wmt_14':['wmt_14:language_pair=cs-en,',
                                'wmt_14:language_pair=de-en,',
                                'wmt_14:language_pair=fr-en,',
                                'wmt_14:language_pair=hi-en,',
                                'wmt_14:language_pair=ru-en,']}

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

class KL:
    
    '''
    Model to estimate the DKL using the classification approach to density ratio estimation
    (this is class in Scikit-Learn style)
    '''
    
    def __init__(self, boost=True, validation_split=.1, cat_features=None, cv=5):
        
        '''
        Input:  (i)   boost: if TRUE, we use CatBoost as classifier - otherwise, we use logistic regression;
                (ii)  validation_split: portion of the training data (Zs,Zt) used to early stop CatBoost - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validate the logistic regression classifier - this parameter is not used if 'boost'==TRUE. Hyperparameter values tested are specified in Scikit-Learn's "LogisticRegressionCV" class. If cv==None, then we use the default Scikit-Learn config. for LogisticRegression;
        '''
        
        self.boost=boost
        self.validation_split=validation_split
        self.cat_features=cat_features
        self.cv=cv
  
    def fit(self, Zs, Zt, random_state=0):
        
        '''
        Function that fits the classification model in order to estimate the density ratio w=p_t/p_s (target dist. over source dist.)

        Input:  (i)   Zs: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution - use the 'prep_data' function to prepare your data;
                (ii)  Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution - use the 'prep_data' function to prepare your data;
                (iii) random_state: seed used in the data splitting and model training
        Output: None
        '''
        
        self.nt, self.ns = Zt.shape[0], Zs.shape[0]
        
        Xw = pd.concat([Zt,Zs], axis=0) 
        yw = np.hstack((np.ones(self.nt),np.zeros(self.ns)))
        
        if self.boost: 
            Xw_train, Xw_val, yw_train, yw_val = train_test_split(Xw, yw, test_size=self.validation_split, random_state=random_state)

            self.model =  CatBoostClassifier(loss_function = 'Logloss',
                                             cat_features=self.cat_features,
                                             thread_count=-1,
                                             random_seed=random_state)

            self.model.fit(Xw_train, yw_train,
                           verbose=False,
                           eval_set=(Xw_val, yw_val),
                           early_stopping_rounds = 100)
         
        else:           
            if self.cv==None:
                self.model = LogisticRegression(solver='liblinear', random_state=random_state).fit(Xw, yw)
            else: 
                self.model = LogisticRegressionCV(cv=self.cv, scoring='neg_log_loss', solver='liblinear', 
                                                  random_state=random_state).fit(Xw, yw)
                
            

    def predict_w(self, Z, eps=10**-10):
        
        '''
        Function that predicts the density ratio w=p_t/p_s (target dist. over source dist.)

        Input:  (i) Z: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution;
        
        Output: (ii) An array containing the predicted density ratio w=p_t/p_s for each row of Z
        '''
        
        p = self.model.predict_proba(Z)[:,1]
        prior_ratio = self.ns/self.nt
        return prior_ratio*((p+eps)/(1-p+eps))

    def predict(self, Zt, eps=10**-10):
        
        '''
        Function that infers the DKL of the distirbutions that generated Zs and Zt

        Input:  (i) Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution;
        
        Output: (i) Point estimate of DKL
        '''
        
        predt=self.predict_w(Zt, eps)
        
        return np.mean(np.log(predt)) 