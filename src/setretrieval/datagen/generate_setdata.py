from ..inference.oai_request_client import ParallelResponsesClient, PRICING
from ..utils.constants import categorize_prompt, conceptual_rephrase_prompt, abstract_questions_prompt, example_abstract_passage, example_abstract_questions
from ..utils.constants import decomposed_prompt_restrictive_4B, decomposed_prompt_restrictive_8B, decomposed_prompt_restrictive_oai
from collections import Counter
from datasets import Dataset
from ..inference.vllm_wrapper import VLLMWrapper
import os
from ..utils.utils import pickload, pickdump
from tqdm import tqdm

# given a pre-processed list of passages with 'text' column, generate abstract questions for each passage.
def passages_to_questions(passages, model="gemini-2.5-pro", pfunct=lambda x: abstract_questions_prompt.format(example_abstract_passage, example_abstract_questions, x)):
    oai_request_client = ParallelResponsesClient(max_concurrent=100)

    # get abstract questions with the prompt
    passagetexts = [passage['text'].replace("\n", " ") for passage in passages]
    abstract_questions_prompts = [pfunct(passage) for passage in passagetexts]
    abstract_questions_responses = oai_request_client.run(model=model, prompts=abstract_questions_prompts)
    
    results = []
    for i, passage in enumerate(passages):
        resp = abstract_questions_responses[i]['response']
        passage['questions'] = [] if resp is None else resp.split("\n")
        passage['cost'] = abstract_questions_responses[i]['cost_usd']
        results.append({
            'text': passage['text'],
            'questions': passage['questions'],
            'cost': passage['cost']
        })
    breakpoint()
    return Dataset.from_list(results)

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
    if os.path.exists(cache):
        # breakpoint()
        allresponses = pickload(cache)
        # breakpoint()
        # some fault tolerance
        if len(allresponses) > len(prior_positives)-2:
            newresps = []
            for i in range(min(len(allresponses), len(prior_positives))):
                if len(prior_positives[i]) == 0:
                    newresps.append([])
                else:
                    newresps.append([prior_positives[i][j] for j in range(len(prior_positives[i])) if allresponses[i][j]])
            return newresps, 0
    else:
        allresponses = []
    
    use_oai = model in PRICING.keys()
    if use_oai:
        print(f"Using OAI model: {model}")
        # use oai model
        oai_client = ParallelResponsesClient(max_concurrent=100, openai_key_path="/accounts/projects/sewonm/prasann/oaikey.sh")
        proc_fct = lambda x: oai_client.run(model=model, prompts=x)
    else:
        print(f"Using VLLM model: {model}")
        # use vllm model, TODO maybe don't need datastore reasoning
        vllm_wrapper = VLLMWrapper(model)
        proc_fct = lambda x: vllm_wrapper.generate(x)
    
    
    print("Mean length of prior positives: ", sum([len(pos) for pos in prior_positives])/len(prior_positives))

    if "Qwen/Qwen3-8B" in model:
        prompt = decomposed_prompt_restrictive_8B
    elif "Qwen/Qwen3-4B" in model:
        prompt = decomposed_prompt_restrictive_4B
    else:
        prompt = decomposed_prompt_restrictive_oai

    if len(prior_positives) != len(queries): 
        print("WARNING POSITIVES / QUERY COUNTS DO NOT MATCH")

    if use_oai:
        # for oai, put all queries together into one big list, then group them together later
        allprompts = []
        origlens = [len(prior_positives[i]) for i in range(len(prior_positives))]
        for i in tqdm(range(len(queries)), desc="Formatting prompts"):
            prior_posvals = list(prior_positives[i])
            prompts = [prompt.format(doc, queries[i]) for doc in prior_posvals]
            allprompts.extend(prompts)
        responses = []
        for i in range(0, len(allprompts), 5000):
            responses.extend(proc_fct(allprompts[i:i+5000]))
            print("Cost so far: ", oai_client.total_cost)
        responses = [response['response'] for response in responses]
        responses = [response.lower().strip() == "yes" if response is not None else False for response in responses]
        allresponses = []
        ind = 0
        for i in range(len(origlens)):
            allresponses.append(responses[ind:ind+origlens[i]])
            pickdump(allresponses, cache)
            ind += origlens[i]
        
    else:
        for i in tqdm(range(len(allresponses), len(prior_positives))):
            prior_posvals = list(prior_positives[i])
            prompts = [prompt.format(doc, queries[i]) for doc in prior_posvals]
            responses = proc_fct(prompts)
            responses = ["yes" in response.outputs[0].text.lower().strip() for response in responses]
            print("Num queries relevant to passage: ", sum(responses))
            allresponses.append(responses)
            pickdump(allresponses, cache)


    cost = 0 if use_oai==False else oai_client.total_cost
    print(f"Total cost: {cost}")

    # breakpoint()
    # convert to list of positive passages
    allresponses = [prior_positives[i][j] for i in range(len(prior_positives)) for j in range(len(prior_positives[i])) if allresponses[i][j]]

    # for each stage, return list of passages relevant to the query
    if not use_oai:
        vllm_wrapper.delete_model()
    return allresponses, cost

def chunks_to_inds(clist):
    # given list of booleans, return indices where True
    return [i for i, c in enumerate(clist) if c]

def hierarchical_positive_search(passages, questions, cachekey, models=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "gemini-2.5-flash", "gemini-2.5-pro"], actualpassages=None, cache="propercache/cache/gendata/"):
    # initialize loop, in gutenberg / other custom cases we can also pass in large loops to begin with
    if type(passages[0]) is not list:
        print("Using the same passage set for all questions.")
        previouspasses = [passages for _ in questions]
    else:
        previouspasses = passages
    
    if actualpassages is None:
        actualpassages = passages
    # map passage to index
    pass2ind = {passage: i for i, passage in enumerate(tqdm(actualpassages, desc="Mapping passages to indices"))}
    # load cache if it exists
    # if os.path.exists(cache+f"{cachekey}.pkl"):
    #     selectresults = pickload(cache+f"{cachekey}.pkl")
    # else:
    selectresults = {}
    for i in range(len(models)):
        # breakpoint()
        if models[i] in selectresults:
            previouspasses = [actualpassages[ind] for ind in selectresults[models[i]]['keep_indices']]
            cost = selectresults[models[i]]['cost']
        else:
            currentpasses, cost = possearch_singlestage(previouspasses, questions, models[i], cache+f"{cachekey}_{i}.pkl")
            print(f"Stage {i} cost: {cost}")
            # breakpoint()
            # save to use later
            curinds = [[pass2ind[passage] if len(passage) > 10 else -1 for passage in currentpasses[j]] for j in range(len(currentpasses))]
            selectresults[models[i]] = {'keep_indices': curinds, 'cost': cost}
            previouspasses = currentpasses
            pickdump(selectresults, cache+f"{cachekey}.pkl")
    return selectresults