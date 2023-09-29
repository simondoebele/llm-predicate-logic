import nltk
from nltk.sem.evaluate import Valuation, Model
from nltk.sem.logic import *
from nltk.sem.logic import LogicParser
import itertools
from itertools import product
import random
import pandas as pd


def generate_unary_predicate_expressions(unary_predicate_tokens, variables):
  """
  Generates predicate Expressions according to the nltk.sem.logic package.
  input: 
  - variables: a list of (lower case) (nltk) Variables as per the nltk.sem.logic package.
  - unary_predicate_tokens: a list of (upper case) (nltk) Variables that represents unary predicates.
  """
  predicate_expressions = []
  predicates = []
  vars = []
  operators = []
  quantifiers = []

  # Atomic formulas (Fx, Gx, ...)
  for upt in unary_predicate_tokens:
    for variable in variables:
      predicate_expressions.append(ApplicationExpression(FunctionVariableExpression(upt), AbstractVariableExpression(variable)))
      predicates.append(set(str(upt)))
      vars.append(set(str(variable)))
      operators.append(set())
      quantifiers.append(set())
  
  # Negated atomic formulas (-Fx, -Gx, ...)
  negated_formulas = []
  for i, pred_expr in enumerate(predicate_expressions):
    negated_formulas.append(NegatedExpression(pred_expr))
    predicates.append(predicates[i])
    vars.append(vars[i])
    operators.append(set(Tokens.NOT))
    quantifiers.append(set())

  predicate_expressions = predicate_expressions + negated_formulas

  return predicate_expressions, predicates, vars, operators, quantifiers



def generate_binary_predicate_expressions(binary_predicate_tokens, variables):
#   """
#   Generates predicate Expressions according to the nltk.sem.logic package.
#   input: 
#   - variables: a list of Variables as per the nltk.sem.logic package.
#   - binary_predicate_tokens: a list of tokens (strings) that represents binary predicates.
#   """
    return NotImplementedError



def generate_complex_predicate_expressions(predicate_expressions, predicates, vars, operators, quantifiers, recursion_depth):
  all_complex_formulas = []

  if recursion_depth >= 1:
    atomic_formulas = predicate_expressions
    all_complex_formulas.extend(atomic_formulas)

  if recursion_depth >= 2:
    level2_formulas = []
    
    for i, antecedent in enumerate(atomic_formulas):
      for j, consequent in enumerate(atomic_formulas):

        # IMP expressions
        level2_formulas.append(ImpExpression(antecedent, consequent))
        predicates.append(predicates[i] | predicates[j])
        vars.append(vars[i] | vars[j])
        operators.append(set(Tokens.IMP) | operators[i] | operators[j])
        quantifiers.append(set())

        # negated IMP expressions:
        level2_formulas.append(NegatedExpression(ImpExpression(antecedent, consequent)))
        predicates.append(predicates[i] | predicates[j])
        vars.append(vars[i] | vars[j])
        operators.append(set([Tokens.IMP,Tokens.NOT]))
        quantifiers.append(set())
        
        # AND expressions:
        level2_formulas.append(AndExpression(antecedent, consequent))
        predicates.append(predicates[i] | predicates[j])
        vars.append(vars[i] | vars[j])
        operators.append(set(Tokens.AND) | operators[i] | operators[j])
        quantifiers.append(set())

        # negated AND expressions:
        level2_formulas.append(NegatedExpression(AndExpression(antecedent, consequent)))
        predicates.append(predicates[i] | predicates[j])
        vars.append(vars[i] | vars[j])
        operators.append(set([Tokens.AND,Tokens.NOT]))
        quantifiers.append(set())

        # add here for more types of expressions (OrExpression, etc.) ...
    
    all_complex_formulas.extend(level2_formulas)

  if recursion_depth >= 3:
    level3_formulas = []
    start_after_atomic = len(atomic_formulas)

    for i, antecedent in enumerate(atomic_formulas):
      for j, consequent in enumerate(level2_formulas):
        # IMP expressions
        level3_formulas.append(ImpExpression(antecedent, consequent))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set(Tokens.IMP) | operators[i] | operators[start_after_atomic + j])
        quantifiers.append(set())

        # negated IMP expressions:
        level3_formulas.append(NegatedExpression(ImpExpression(antecedent, consequent)))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set([Tokens.IMP,Tokens.NOT]))
        quantifiers.append(set())

        # IMP expressions REVERSE DIRECTION
        level3_formulas.append(ImpExpression(consequent, antecedent))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set(Tokens.IMP) | operators[i] | operators[start_after_atomic + j])
        quantifiers.append(set())

        # negated IMP expressions: REVERSE DIRECTION
        level3_formulas.append(NegatedExpression(ImpExpression(consequent, antecedent)))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set([Tokens.IMP,Tokens.NOT]))
        quantifiers.append(set())
        
        # AND expressions:
        level3_formulas.append(AndExpression(antecedent, consequent))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set(Tokens.AND) | operators[i] | operators[start_after_atomic + j])
        quantifiers.append(set())

        # negated AND expressions:
        level3_formulas.append(NegatedExpression(AndExpression(antecedent, consequent)))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set([Tokens.AND,Tokens.NOT]))
        quantifiers.append(set())

        # AND expressions: REVERSE DIRECTION
        level3_formulas.append(AndExpression(consequent, antecedent))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set(Tokens.AND) | operators[i] | operators[start_after_atomic + j])
        quantifiers.append(set())

        # negated AND expressions: REVERSE DIRECTION
        level3_formulas.append(NegatedExpression(AndExpression(consequent, antecedent)))
        predicates.append(predicates[i] | predicates[start_after_atomic + j])
        vars.append(vars[i] | vars[start_after_atomic + j])
        operators.append(set([Tokens.AND,Tokens.NOT]))
        quantifiers.append(set())

        # add here for more types of expressions (OrExpression, etc.) ...
    all_complex_formulas.extend(level3_formulas)
  if recursion_depth >= 4:
    NotImplementedError
    
  return all_complex_formulas, predicates, vars, operators, quantifiers
    



def generate_quantified(complex_formulas, predicates, vars, operators, quantifiers, num_quantifiers = 1):

  quantified_formulas = []

  if num_quantifiers == 1:
    for i, formula in enumerate(complex_formulas):
      for var in vars[i]:
        
        # All-Quantified:
        quantified_formulas.append(AllExpression(Variable(var), formula))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i])
        quantifiers.append(Tokens.ALL)

        # Negation of All-Quantified:
        quantified_formulas.append(NegatedExpression(AllExpression(Variable(var), formula)))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i] | set(Tokens.NOT))
        quantifiers.append(Tokens.ALL)

        # Exists-Quantified:
        quantified_formulas.append(ExistsExpression(Variable(var), formula))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i])
        quantifiers.append(Tokens.EXISTS)

        # Negation of Exists Quantified: 
        quantified_formulas.append(NegatedExpression(ExistsExpression(Variable(var), formula)))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i] | set(Tokens.NOT))
        quantifiers.append(Tokens.EXISTS)

  if num_quantifiers == 2:
    for i, formula in enumerate(complex_formulas):
      if len(vars[i]) == 2:
        vars_lst = list(vars[i])
        x = vars_lst[0]
        y = vars_lst[1]
        
        # All-Exists-Quantified:
        quantified_formulas.append(AllExpression(Variable(x), ExistsExpression(Variable(y), formula)))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i])
        quantifiers.append([Tokens.ALL, Tokens.EXISTS])

        # Negation of All-Exists-Quantified:
        quantified_formulas.append(NegatedExpression(AllExpression(Variable(x), ExistsExpression(Variable(y), formula))))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i] | set(Tokens.NOT))
        quantifiers.append([Tokens.ALL, Tokens.EXISTS])

        # Exists-All-Quantified:
        quantified_formulas.append(ExistsExpression(Variable(x), AllExpression(Variable(y), formula)))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i])
        quantifiers.append([Tokens.ALL, Tokens.EXISTS])

        # Negation of Exists-All-Quantified: 
        quantified_formulas.append(NegatedExpression(ExistsExpression(Variable(x), AllExpression(Variable(y), formula))))
        predicates.append(predicates[i])
        vars.append(vars[i])
        operators.append(operators[i] | set(Tokens.NOT))
        quantifiers.append([Tokens.ALL, Tokens.EXISTS])

  if True: # decide to cutoff complex_formulas, as for the nltk theorem prover, they always need a quantifier (given that they have a variable such as x, and not just a constant.)
      cutoff = len(complex_formulas)
      predicates = predicates[cutoff:]
      vars = vars[cutoff:]
      operators = operators[cutoff:]
      quantifiers = quantifiers[cutoff:]

  return quantified_formulas, predicates, vars, operators, quantifiers



def map_keys(predicates, adjectives, names, possible_constant_tokens, max_num_names=10, num_distractors=0):
  # TODO: potential add-on: add distractors.

  # Map predicates to adjectives
  predicate_mapping = {}
  random.shuffle(adjectives)
  num_predicates = len(predicates)
  # Take the first few as the random subset
  adjectives_subset = adjectives[:num_predicates]
  # (unique mapping!)
  for i, predicate in enumerate(predicates):
      predicate_mapping[predicate] = adjectives_subset[i]

  # Map variable-constants to names
  constant_mapping = {}
  num_names = random.randint(1, max_num_names)
  random_names = random.sample(names, num_names)
  random.shuffle(possible_constant_tokens)
  constants_subset = possible_constant_tokens[:num_names]
  # (unique mapping!)
  for i, constant in enumerate(constants_subset):
      constant_mapping[constant] = random_names[i]
          
  return predicate_mapping, constant_mapping, random_names, adjectives_subset



# NOTE: #Possible TODO: generalise for binary, ternary,... predicates. 
def create_world_model(predicate_mapping, constant_mapping, negation):
  # natural language representation of the model
  sentences = ""
  # nltk representation of the model
  expressions = []

  for predicate, adjective in predicate_mapping.items():
    for constant, name in constant_mapping.items():
      include_negation = random.choice([True, False])
      if include_negation:
        sentence = name + " is " + negation + " " + adjective + "."
        # for negation, we do not add to the expressions, as this is implicit knowledge (closed-world assumption)
      else:
        sentence = name + " is " + adjective + "."
        #expressions.append(ApplicationExpression(FunctionVariableExpression(Variable(predicate)), ConstantExpression(Variable(constant))))
        expressions.append((predicate, set([constant])))
      sentences = sentences + " " + sentence

  return sentences, expressions



# helper function: that converts predicate and constant mappings into string format for storage.
def get_keys(predicate_mapping, constant_mapping):
  both = predicate_mapping | constant_mapping
  keys = ' '.join([f"{key}: {value}." for key, value in both.items()])
  return keys



def make_task1_inputs_prompt1(df):
    keys_list = df['Keys'].tolist()
    world_models = df['World Model'].tolist()
    sat_list = df['Satisfied'].tolist()
    #satisfied_list = ['satisfied' if x == "sati" else 'unsatisfied' for x in sat_list]

    p1 = "Here is a world model: "
    p2 = " Let us interpret predicates and names as follows: "
    p3 = ". Provide a formula in first-order predicate logic that is "
    p4 = " given the above situation and interpretation (keys)."

    prompts = []
    for keys, model, sat in zip(keys_list, world_models, sat_list):
      prompt = p1 + model + p2 + keys + p3 + sat + p4
      prompts.append(prompt)
    
    return prompts




def make_task2_inputs_prompt1(df):
    formulas = df['Formula'].tolist()
    sat_list = df['Satisfied'].tolist()

    p1 = "Consider the following formula: "
    p2 = ". Describe a situation in which the formula is "
    p3 = " and provide the keys."

    prompts = []
    for formula, sat in zip(formulas, sat_list):
      prompt = p1 + formula + p2 + sat + p3
      prompts.append(prompt)
      sat_list.append(sat) 
    
    return prompts




def make_task3_inputs_prompt1(df):
    formulas = df['Formula'].tolist()
    keys_list = df['Keys'].tolist()
    world_models = df['World Model'].tolist()

    p1 = "Consider the following formula in first-order predicate logic: "
    p2 = ". Let us interpret predicates and names as follows: "
    p3 = " Also, here is a world model: "
    p4 = " Is the provided formula satisfied or not given the sitation? ('satisfied' or 'unsatisfied?')."

    prompts = []
    unique_ids = [] # to be changed (and returned), if we make these datasets a different size than the original one that we map to.
    for i, (formula, keys, model) in enumerate(zip(formulas, keys_list, world_models)):
      prompt = p1 + formula + p2 + keys + p3 + model + p4
      prompts.append(prompt)
      unique_ids.append(i)
    
    return prompts


def make_task1_inputs(df):
    keys_list = df['Keys'].tolist()
    world_models = df['World Model'].tolist()
    sat_list = df['Satisfied'].tolist()

    context = "Context:\n\nYou will receive a problem in first-order predicate logic to solve. \
        This problem contains a set of statements about the world (let us call it a 'world model') \
        and a mapping from things in this world to variables representing these things \
        (we call that mapping the 'keys'). You are then asked to provide a formula in first-order predicate logic \
        that is either satisfied or unsatisfied given the world model and keys.\
        Please return only the formula, written in the format of the python library nltk.\n\n"
    instruction_p1 = "Question:\n\n Here is a world model:\n\n"
    instruction_p2 = "\n\nHere are the keys:\n\n"
    question_p1 = "\n\nPlease write down only one formula in first-order predicate logic that is " 
    question_p2 = " given the above world model and keys.\n\n"
    answer = "Answer:"

    prompts = []
    for keys, model, sat in zip(keys_list, world_models, sat_list):
      prompt = context + instruction_p1 + model + instruction_p2 + keys + question_p1 + sat + question_p2 + answer
      prompts.append(prompt)
    
    return prompts



def make_task2_inputs(df):
    formulas = df['Formula'].tolist()
    sat_list = df['Satisfied'].tolist()

    context = "Context:\n\nYou will receive a problem in first-order predicate logic to solve. \
      This problem contains a formula in first-order predicate logic, written in the format of the python library nltk \
      for which you are asked to create a 'world model' and 'keys' that either satisfy or do not satisfy the formula. \
      A world model is a set of statements about whether one or more predicates apply to things in this world.\
      Keys are a mapping from things to (lower case) variables and predicates to (upper case) variables.\n\n"
    instruction_p1 = "Question:\n\nConsider the following formula:"
    instruction_p2 = ". Provide a world model and keys such that the formula is "
    instruction_p3 = ". Please answer by returning the world model starting with 'World model:', followed by the keys starting with 'Keys:'." 
    answer = "\n\nAnswer:"

    prompts = []
    for formula, sat in zip(formulas, sat_list):
      prompt = context + instruction_p1 + formula + instruction_p2 + sat + instruction_p3 + answer
      prompts.append(prompt)
      sat_list.append(sat) 
    
    return prompts


def make_task3_inputs(df):
    formulas = df['Formula'].tolist()
    keys_list = df['Keys'].tolist()
    world_models = df['World Model'].tolist()

    context = "Context:\n\nYou will receive a problem in first-order predicate logic to solve. \
      This problem contains a formula in first-order predicate logic, written in the format of the python library nltk \
      a set of statements about a world ('world model') and a mapping ('keys') from things in this world to variables \
      representing these things. Please return in one word whether the formula is satisfied or unsatisfied given the world model and keys.\n\n"
    instruction_p1 = "Question:\n\nConsider the following formula:"
    instruction_p2 = "\n\nHere is a world model:\n\n"
    instruction_p3 = "\n\nHere are the keys:\n\n"
    instruction_p4 = "Is the provided formula satisfied or unsatisfied given the world model and keys?"
    answer = "\n\nAnswer:"

    prompts = []
    unique_ids = [] # to be changed (and returned), if we make these datasets a different size than the original one that we map to.
    for i, (formula, keys, model) in enumerate(zip(formulas, keys_list, world_models)):
      prompt = context + instruction_p1 + formula + instruction_p2 + model + instruction_p3 + keys + instruction_p4 + answer
      prompts.append(prompt)
      unique_ids.append(i)
    
    return prompts