# script to play around with GEPA a little bit
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
# parallelresponsesclient
from setretrieval.inference.oai_request_client import ParallelResponsesClient
import nltk
from statistics import mean
nltk.download('averaged_perceptron_tagger_eng')

def score(resultstr: str) -> float:
    nouns = nltk.pos_tag(nltk.word_tokenize(resultstr))
    nouns = [word for word, pos in nouns if pos == 'NN']
    return len(set(nouns))

def evaluate(candidate: str) -> float:
    """Score a candidate and log diagnostics as ASI."""
    # breakpoint()
    constraintlist = ["Talk about jungles.", "Talk about humor.", "Talk about animals."]
    prompts = [candidate + " Constraint: " + constraint for constraint in constraintlist]
    breakpoint()
    result = ParallelResponsesClient().run(model="gemini-2.5-flash-lite", prompts=prompts)
    
    resultscore = [score(r['response']) for r in result]
    for r, s in zip(result, resultscore):
        resultstr = "Generated poem: " + r['response'] + "\n" + "Score: " + str(s)
        oa.log(f"Output: {resultstr}")
    breakpoint()
    return mean(resultscore)

if __name__ == "__main__":
    result = optimize_anything(
        seed_candidate="Given the following constraint, generate a poem.",
        evaluator=evaluate,
        objective="I have a secret reward function for generated poems given constraints (the reward function is the same indepedent of provided constraints). I want to figure out a prompt that will maximize the reward function (should generalize to any constraint).",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
    )

    print("Best candidate:", result.best_candidate)