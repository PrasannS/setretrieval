from ..inference.oai_request_client import ParallelResponsesClient, PRICING
from ..utils.constants import categorize_prompt, conceptual_rephrase_prompt, abstract_questions_prompt, example_abstract_passage, example_abstract_questions, decomposed_prompt_restrictive_v2
from collections import Counter
from datasets import Dataset
from ..inference.vllm_wrapper import VLLMWrapper
import os
from ..utils.utils import pickload, pickdump
from tqdm import tqdm

# given a pre-processed list of passages with 'text' column, generate abstract questions for each passage.
def passages_to_questions(passages, model="gemini-2.5-pro"):
    oai_request_client = ParallelResponsesClient(max_concurrent=100)

    # get abstract questions with the prompt
    passagetexts = [passage['text'].replace("\n", " ") for passage in passages]
    abstract_questions_prompts = [abstract_questions_prompt.format(example_abstract_passage, example_abstract_questions, passage) for passage in passagetexts]
    abstract_questions_responses = oai_request_client.run(model=model, prompts=abstract_questions_prompts)
    
    for i, passage in enumerate(passages):
        resp = abstract_questions_responses[i]['response']
        passage['questions'] = [] if resp is None else resp.split("\n")
        passage['cost'] = abstract_questions_responses[i]['cost_usd']

    return passages

# Take in querydata (dataset with 'query' and 'pos' and 'neg' columns)
# Return dataset with more natural questions, as well as some topic categorizations
# NOTE -> see prompts used in constants.py
def categorize_filter_setdata(querydata, model="gemini-2.5-flash", limit=30000):
    oai_client = ParallelResponsesClient(max_concurrent=100)

    questions = [row['query'] for row in querydata[:limit]]
    category_prompts = [categorize_prompt.format(question) for question in questions]
    category_responses = oai_client.run(model=model, prompts=category_prompts)
    
    # count category counts
    category_counts = Counter([r['response'] for r in category_responses])
    print(category_counts)
    print(oai_client.total_cost)

    # do rephrasing to be more "natural" / "general"
    conceptual_rephrase_prompts = [conceptual_rephrase_prompt.format(question) for question in questions]
    conceptual_rephrase_responses = oai_client.run(model=model, prompts=conceptual_rephrase_prompts)
    assert len(conceptual_rephrase_responses) == len(category_responses)

    # create new dataset with more natural questions, as well as some topic categorizations
    newdata = []
    for i in range(len(conceptual_rephrase_responses)):
        newdata.append({
            'query': questions[i],
            'pos': querydata[i]['pos'],
            'neg': querydata[i]['neg'],
            'category': category_responses[i]['response'],
            'rephrased_query': conceptual_rephrase_responses[i]['response']
        })
    
    return Dataset.from_list(newdata)

# def possearch_singlestage(prior_positives, queries, model):

#     use_oai = model in PRICING.keys()
#     if use_oai:
#         print(f"Using OAI model: {model}")
#         # use oai model
#         oai_client = ParallelResponsesClient(max_concurrent=100)
#         proc_fct = lambda x: oai_client.run(model=model, prompts=x)
#     else:
#         print(f"Using VLLM model: {model}")
#         # use vllm model, TODO maybe don't need datastore reasoning
#         vllm_wrapper = VLLMWrapper(model)
#         proc_fct = lambda x: vllm_wrapper.generate(x)
        
#     # get a list of all the queries in one go
#     pos_lens = [len(pos) for pos in prior_positives]
#     queries_flat = []
#     pos_flat = []
#     for i in range(len(prior_positives)):
#         pos_flat.extend(prior_positives[i])
#         queries_flat.extend([queries[i]]*len(prior_positives[i]))
#     queries_all = [decomposed_prompt_restrictive_v2.format(query, pos) for pos in pos_flat for query, pos in zip(queries_flat, pos_flat)]

#     responses = proc_fct(queries_all)
#     responses = ["yes" in response['response'].lower().strip() for response in responses]
#     cost = 0 if use_oai else vllm_wrapper.total_cost
#     print(f"Total cost: {cost}")
#     responses_grouped = []
#     ind = 0
#     # for each stage, return list of passages relevant to the query
#     for i in range(len(pos_lens)):
#         responses_grouped.append([prior_positives[i][j] for j in range(len(prior_positives[i])) if responses[ind+j]])
#         ind += pos_lens[i]
#     if not use_oai:
#         vllm_wrapper.delete_model()
#     return responses_grouped, cost

def possearch_singlestage(prior_positives, queries, model, cache):

    # breakpoint()
    # only use the first one for now
    
    use_oai = model in PRICING.keys()
    if use_oai:
        print(f"Using OAI model: {model}")
        # use oai model
        oai_client = ParallelResponsesClient(max_concurrent=100)
        proc_fct = lambda x: oai_client.run(model=model, prompts=x)
    else:
        print(f"Using VLLM model: {model}")
        # use vllm model, TODO maybe don't need datastore reasoning
        vllm_wrapper = VLLMWrapper(model)
        proc_fct = lambda x: vllm_wrapper.generate(x)
    
    prior_posvals = list(prior_positives[0])
    
    allresponses = []
    for i in tqdm(range(len(queries))):
        prompts = [decomposed_prompt_restrictive_v2.format(queries[i], doc) for doc in prior_posvals]
        responses = proc_fct(prompts)
        responses = ["yes" in response.outputs[0].text.lower().strip() for response in responses]
        print("Num queries relevant to passage: ", sum(responses))
        allresponses.append(responses)
        pickdump(allresponses, cache)


    cost = 0 if use_oai else vllm_wrapper.total_cost
    print(f"Total cost: {cost}")
    responses_grouped = []
    # for each stage, return list of passages relevant to the query
    if not use_oai:
        vllm_wrapper.delete_model()
    return responses_grouped, cost

def hierarchical_positive_search(passages, questions, cachekey, models=["Qwen/Qwen3-4B", "gemini-2.5-flash", "gemini-2.5-pro"], cache="cache/gendata/"):
    # initialize loop, in gutenberg / other custom cases we can also pass in large loops to begin with
    if type(passages[0]) is not list:
        print("Using the same passage set for all questions.")
        previouspasses = [passages for _ in questions]
    
    # map passage to index
    pass2ind = {passage: i for i, passage in enumerate(tqdm(passages, desc="Mapping passages to indices"))}
    # load cache if it exists
    if os.path.exists(cache+f"{cachekey}.pkl"):
        selectresults = pickload(cache+f"{cachekey}.pkl")
    else:
        selectresults = {}
    for i in range(len(models)):
        if models[i] in selectresults:
            previouspasses = [passages[ind] for ind in selectresults[models[i]]['keep_indices']]
            cost = selectresults[models[i]]['cost']
        else:
            currentpasses, cost = possearch_singlestage(previouspasses, questions, models[i], cache+f"{cachekey}_{i}.pkl")
            print(f"Stage {i} cost: {cost}")
            # save to use later
            curinds = [[pass2ind[passage] for passage in currentpasses[j]] for j in range(len(currentpasses))]
            selectresults[models[i]] = {'keep_indices': curinds, 'cost': cost}
            previouspasses = currentpasses
            pickdump(selectresults, cache+f"{cachekey}.pkl")
    return selectresults