from nltk.sem.evaluate import Valuation, Model
from nltk.sem.logic import *
from nltk.sem.logic import LogicParser
import nltk
import re



def parse_LLM_output(output):
    constant_mapping = []
    nltk_expressions = []

    output = output.strip() # strips whitespace in beginning and end.

    # EITHER the LLM specifies which is the world model and keys...
    if re.search('keys', output, re.IGNORECASE) and re.search('world model', output, re.IGNORECASE):
        output = re.split(r'keys', output, flags=re.IGNORECASE)

        # KEYS:
        keys_str = output[1] # NOTE: we ignore any second naming of "keys" (which could place also at output[2], etc.)
        keys_list = re.split('\\.', keys_str, flags=re.IGNORECASE)

        # WORLD_MODEL:
        world_model = output[0]
        # clean the string
        world_model = re.sub('world model', '', world_model, flags=re.IGNORECASE)
        world_model = world_model.lstrip("\\:")
        world_model = world_model.lstrip()

        world_model_list = re.split('\\.', world_model, flags=re.IGNORECASE)
        world_model_list = [x.strip() for x in world_model_list if x]

    # OR we need to find it ourselves.
    else:
        output = re.split('\\.', output)
        world_model_list = []
        keys_list = []
        for sentence in output:
            if "is" in sentence:
                world_model_list.append(sentence)
            if ":" in sentence:
                keys_list.append(sentence)

    reverse_predicate_mapping = {}
    reverse_constant_mapping = {}
    
    for k in keys_list:
        if len(k) > 0: # reason for this: splitting at the last "." leads to an empty string
            # clean the string (e.g. ":" after "keys:..." as well as white space after ":")
            k = k.lstrip("\\:")
            kv = k.split(":") # split keys and values
            kv = [i.lstrip() for i in kv] # clean whitespace
            if len(kv) > 1:
                if len(kv[0]) == 1 and kv[0].isupper(): #check for uppercase
                    reverse_predicate_mapping[kv[1]] = kv[0] # e.g. Anxious: F. instead of: F: Anxious.
                elif len(kv[0]) == 1 and kv[0].islower():
                    reverse_constant_mapping[kv[1]] = kv[0]
    if len(reverse_constant_mapping) > 0:
        constant_mapping = [(v, k) for k, v in reverse_constant_mapping.items()]

    for sentence in world_model_list:
        # split sentence into words
        sentence = sentence.split()
        # get indices before and after is
        # here, we require that the LLM makes well-formed sentences (i.e. subject - verb "is" - [not] - adj). Otherwise, we do not take it).
        if "is" in sentence and sentence.index("is") > 0: 
            is_index = sentence.index("is")
            if "not" in sentence:
                if sentence.index("not") == is_index + 1: # again: needs to be grammatically correct
                    try: # we may not have a reverse_predicate_mapping or reverse_constant_mapping, if keys are missing
                        name = sentence[is_index - 1] 
                        adjective = sentence[is_index + 2]
                        predicate = reverse_predicate_mapping[adjective] # mapping zum uppercase letter nötig!
                        constant = reverse_constant_mapping[name]
                        nltk_expressions.append((predicate, set(constant)))
                    except:
                        pass
            else:
                try: 
                    name = sentence[is_index - 1] 
                    adjective = sentence[is_index + 1]
                    predicate = reverse_predicate_mapping[adjective] # mapping zum uppercase letter nötig!
                    constant = reverse_constant_mapping[name]
                    nltk_expressions.append((predicate, set(constant)))
                except:
                    pass
    return nltk_expressions + constant_mapping