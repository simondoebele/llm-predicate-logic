from Parser import parse_LLM_output
from nltk.sem.logic import *
import nltk
from nltk.sem.logic import LogicParser, Expression
from nltk.sem.evaluate import Valuation, Model


def convert_valuation_back(valuation):
    # this is necessary, as jsonl could not serialize sets, but nltk expects sets for predicates.
    return [(v[0], set(v[1])) if v[0].isupper() else v for v in valuation]


def postprocess_output(output):
    # find the first and longest substring that can be parsed by the nltk parser. Has to include a "." after an exists or all quantifier
    read_expr = Expression.fromstring
    l = len(output)
    start1 = output.find('exists x.')
    start2 = output.find('all x.')
    if start1 != -1 and start2 != -1: # we can only take the smallest value, if it's not -1.
        start = min(start1, start2)
    else:
        start = max(start1, start2)
    #start = min(start1, start2)
    if start == -1:
        gibberishDetected = True
        return "-100", gibberishDetected
    else:
        successfully_parsed = []
        for i in range(start, l, 1):
            try:
                temp = read_expr(output[start:i])
                successfully_parsed.append(temp)
            except:
                pass
        if len(successfully_parsed) > 0:
            gibberishDetected = False
            return successfully_parsed[-1], gibberishDetected # choose the longest formula it can find (substring method)
        else:
            gibberishDetected = True
            return "-100", gibberishDetected # something that is not a formula and will not lead the nltk theorem prover to return either satisfied or unsatisfied.


def eval_task1(dataset):
    correctIncorrect = []
    gibberish = []
    for generated_output, target, valuation in zip(dataset["Predictions"], dataset["Target-sat"], dataset["Valuation"]):
        generated_formula, gibberishDetected = postprocess_output(generated_output)
        gibberish.append(gibberishDetected)
        prediction = "undefined"
        try:
            valuation = convert_valuation_back(valuation)
            val = Valuation(valuation)
            dom = val.domain
            m = nltk.sem.evaluate.Model(dom, val)
            g = nltk.sem.Assignment(dom)
            sat = m.evaluate(str(generated_formula), g)
            if sat == True:
                prediction = "satisfied"
            elif sat == False:
                prediction = "unsatisfied"
        except:
            pass
        
        correctIncorrect.append(prediction==target)

    return correctIncorrect, gibberish


def eval_task2(dataset):
    correctIncorrect = []
    gibberish = []
    for generated_output, target, formula in zip(dataset["Predictions"], dataset["Target-sat"], dataset["Formulas"]):
        v = parse_LLM_output(generated_output)
        if len(v) >= 1:
            gibberish.append(False)
        else:
            gibberish.append(True)

        val = Valuation(v)
        dom = val.domain
        m = nltk.sem.evaluate.Model(dom, val)
        g = nltk.sem.Assignment(dom)
        sat = m.evaluate(formula, g)
        
        if len(v) == 0:
            prediction = "undefined"
        elif sat == True:
            prediction = "satisfied"
        elif sat == False:
            prediction = "unsatisfied"
        else:
            prediction = "undefined"
        
        correctIncorrect.append(prediction==target)

    return correctIncorrect, gibberish


def eval_task3(dataset):
    correctIncorrect = []
    gibberish = []
    for generated_output, target in zip(dataset["Predictions"], dataset["References"]):
        # we search for keywords, as (some) LLMs tend to output more than one word.
        to_search_for = ["satisfied", "not satisfied", "unsatisfied"]
        to_compare = generated_output
        gibberishDetected = True
        for keyword in to_search_for:
            if keyword in generated_output:
                to_compare = keyword
                gibberishDetected = False
        gibberish.append(gibberishDetected)
        correctIncorrect.append(to_compare==target)

    return correctIncorrect, gibberish

    
def getAccuracy(correctIncorrect):
    return sum(correctIncorrect)/len(correctIncorrect)


def getAccuracyIgnoringGibberish(correctIncorrect, gibberish):
    return sum(correctIncorrect)/(len(correctIncorrect) - sum(gibberish))
