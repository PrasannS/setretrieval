### NOTE code for 3 approaches to reasoning with a datastore
# 1. use the book as a whole to answer questions (API LLM)
# 2. use the book as a whole to answer questions (RAG, I guess colbert can be used here since the library is pretty easy)
# 3. upper-bound where LLM is applied to each chunk to generate a note, and then there's an aggregation step
from datasets import Dataset
from tqdm import tqdm
from .easy_indexer import EasyIndexer
from .oai_request_client import ParallelResponsesClient
from vllm import LLM, SamplingParams
import torch    
from typing import Dict, List
import random
from rank_bm25 import BM25Okapi
import numpy as np
import hashlib
from ..utils.constants import decomposed_prompt, decomposed_prompt_restrictive, decomposed_prompt_restrictive_v2, wholebook_prompt, aggregated_prompt

# get book which has "45873" in the id
def get_book_by_id(idstr, blist):
    for b in blist:
        if  b['id'][:-1]==idstr+"-":
            print(b['id'])
            return b
    return None


def naive_alltext_chunk(doc, words_per_chunk=500): 
    # chunk a document into words. Keep track of and return word indices in the document
    words = doc.split()
    for i in range(0, len(words), words_per_chunk):
        yield " ".join(words[i:i + words_per_chunk])

# create a new dataset based on chunking each document up into 500 word chunks. Keep track of index, as well as word indices in the document (multiple of 500)
def make_chunk_dset(docs, words_per_chunk=500):
    chunked_docs = []
    for i, doc in tqdm(enumerate(docs)):
        for j, chunk in enumerate(naive_alltext_chunk(doc['text'], words_per_chunk=words_per_chunk)):
            chunked_docs.append({'text': chunk, 'doc_id': i, 'chunk_id': j, 'start_word': j * words_per_chunk, 'end_word': (j + 1) * words_per_chunk})
    chunked_docs = Dataset.from_list(chunked_docs)
    return chunked_docs

def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def get_deterministic_hash(input_string):
    # Encode the string to bytes (UTF-8 is a common choice)
    encoded_string = input_string.encode('utf-8')

    # Create a SHA256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the encoded string
    sha256_hash.update(encoded_string)

    # Get the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

class DatastoreReasoning:
    def __init__(self, embedding_model="nomic-ai/nomic-embed-text-v1", max_concurrent=25, vllm_model=None):
        # using custom thing
        self.oai_client =  ParallelResponsesClient(max_concurrent=max_concurrent)
        self.global_rag = EasyIndexer(model_name=embedding_model)
        self.embedding_model = embedding_model
        if vllm_model is not None:
            self.vllm_model = LLM(model=vllm_model, tensor_parallel_size=torch.cuda.device_count(), max_model_len=131072, rope_scaling={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768})

    def indivdoc_oai_reasoning(self, docs, question, model="gpt-5-mini", restrictive=False, thinking=False, restrictive_v2=False):
        
        useprompt = decomposed_prompt_restrictive if restrictive else decomposed_prompt_restrictive_v2 if restrictive_v2 else decomposed_prompt
        if restrictive_v2:
            allqueries = [useprompt.format(question, passage['text']) for passage in docs]
        else:
            allqueries = [useprompt.format(passage['text'], question) for passage in docs]
        print(allqueries[0])
        if "gpt" in model:
            # breakpoint()
            responses = self.oai_client.run(model=model, prompts=allqueries, reasoning="minimal")
        else:
            # use qwen model
            toker = self.vllm_model.get_tokenizer()
            allqueries = [toker.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=thinking) for q in allqueries]
            # breakpoint()
            responses = self.vllm_model.generate(allqueries, sampling_params=SamplingParams(max_tokens=1000))
            responses = [{'response': out.outputs[0].text} for out in responses]
        
        return responses

    def process_input(self, input, words_per_chunk=500):
        if isinstance(input[0], Dict):
            passages = make_chunk_dset(input, words_per_chunk=words_per_chunk)
        elif isinstance(input[0], List):
            print("Using pre-computed set of inputs")
            if len(input) > 1:
                return [Dataset.from_list([{'text': item} for item in book]) for book in input]
            passages = Dataset.from_list([{'text': item} for item in input[0]])
        return passages

    def str_input(self, book, question, words_per_chunk=500, truncation_chars=1000000, randomize_context=False):
        # truncate book to at most 1000000 characters
        passages = self.process_input([book], words_per_chunk=words_per_chunk)
        passageinds = list(range(len(passages)))
        if randomize_context:
            random.shuffle(passageinds)
            passages = [passages[i] for i in passageinds]

        def getcleanpassagetext(ind, passage):
            t = passage['text'].replace("\n", " ")
            return f"[Passage {ind}] {t}"

        truncbook = "\n".join([getcleanpassagetext(i, passage) for i, passage in enumerate(passages)])
        print("Truncating book from ", len(truncbook), " characters to 1000000 characters")

        truncbook = truncbook[:truncation_chars]
        # TODO will maybe want to do something to cover remaining tokens as well?

        
        # get prompt with book and a question
        prompt = wholebook_prompt.format(question, truncbook, question)

        # breakpoint()
        return prompt, passageinds
    
    # use the book as a whole to answer questions (API LLM)
    def wholebook_llm_reasoning(self, book, question, model="gpt-5", words_per_chunk=500, randomize_context=False):
        
        prompt, passageinds = self.str_input(book, question, words_per_chunk=words_per_chunk, randomize_context=randomize_context)
        # breakpoint()
        response = self.oai_client.run(model=model, prompts=[prompt], reasoning="medium", max_output_tokens=10000)
        # breakpoint()
        if randomize_context:
            return response[0]['response'],  passageinds
        return response[0]['response']
    
    def wholebook_vllm_reasoning(self, book, question, words_per_chunk=500, randomize_context=False, debug=False, temp=0.8, num_outputs=1, mtoks=1000):
        toker = self.vllm_model.get_tokenizer()

        if isinstance(question, list):
            assert len(question) == len(book), "Question and book must have the same length"
            allprompts = [self.str_input(book[i], question[i], words_per_chunk=words_per_chunk, randomize_context=randomize_context) for i in range(len(book))]
            allpassageinds = [p[1] for p in allprompts]
            allprompts = [p[0] for p in allprompts]
        else:
            prompt = self.str_input(book, question, words_per_chunk=words_per_chunk, randomize_context=randomize_context)
            allprompts = [prompt[0]]
            allpassageinds = [prompt[1]]
        prompts = toker.apply_chat_template([[{"role": "user", "content": p}] for p in allprompts], tokenize=False, add_generation_prompt=True)
        if num_outputs > 1:
            ps = []
            for p in prompts:
                ps.extend([p]*num_outputs)
            prompts = ps
        if debug:
            breakpoint()
        sparams = SamplingParams(max_tokens=10000, temperature=temp) if temp is not None else SamplingParams(max_tokens=10000)
        response = self.vllm_model.generate(prompts, sampling_params=sparams)
        if num_outputs > 1:
            # response group together
            responses = [response[i:i+num_outputs] for i in range(0, len(response), num_outputs)]
            return responses
        if randomize_context:
            return [out.outputs[0].text for out in response], allpassageinds
        
        return [out.outputs[0].text for out in response]

    # break book up into chunks, then use LLM to generate a note for each chunk, then aggregate the notes to answer the question
    def decomposed_llm_reasoning(self, book, question, model="gpt-5-mini", restrictive=False, restrictive_v2=False, words_per_chunk=500, maxpassages=1e9):

        # get prompt with book and a question 
        passages = self.process_input([book], words_per_chunk=words_per_chunk)
        if len(passages) > maxpassages:
            print("Truncating book from ", len(passages), " passages to ", maxpassages, " passages")
            passages = passages.select(range(maxpassages))

        # separate for modularity / testing
        responses = self.indivdoc_oai_reasoning(passages, question, model, restrictive=restrictive, restrictive_v2=restrictive_v2)

        if restrictive or restrictive_v2: 
            relevantdocs = ['yes' in response['response'].lower() for response in responses]
            print(sum(relevantdocs),  " of ", len(passages), " passages are relevant to the query")
            return responses, relevantdocs
        
        # For all the responses where the note is not "NOT RELEVANT", add the note to a list, along with the passage number
        notes = []
        for i, response in enumerate(responses):
            if "NOT RELEVANT" not in response['response']:
                notes.append(f"Passage {i}: {response['response']}")
        print("There are ", len(notes), " notes out of ", len(responses), " passages")
        totresps = len(responses)
        # breakpoint()
        # now aggregate the responses and answer the question
        fullnote = aggregated_prompt.format("\n".join(notes), question)
        response = self.oai_client.run(model=model, prompts=[fullnote], reasoning="low")
        return {"answer": response[0]['response'], "notes": notes, "totresps": totresps}

    # use retrieval to answer the question
    def easyindex_reasoning(self, book, question, idstr, words_per_chunk=500, k=1000):

        passages = self.process_input([book], words_per_chunk=words_per_chunk)
        print("There are ", len(passages), " passages")
        passages_str = [passage['text'] for passage in passages]
        # loading or searching index
        self.global_rag.index_documents(passages_str, index_id=idstr)
        # breakpoint()
        response = self.global_rag.search([question], index_id=idstr, k=min(k, len(passages)))
        return response

    def bm25_reasoning(self, book, question, words_per_chunk=500, k=1000):
        passages = self.process_input([book], words_per_chunk=words_per_chunk)
        passages_str = [passage['text'] for passage in passages]
        tokenized = [passage.split(" ") for passage in passages_str]
        bm25 = BM25Okapi(tokenized)
        query = question.split(" ")
        bm25_scores = bm25.get_scores(query)
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        return [{'text': passages_str[i], 'index': i, 'score': bm25_scores[i]} for i in top_indices]

    def bm25_reasoning_all_easyindex(self, questionlist, idstr, k=1000):
        assert self.global_rag.index_exists(idstr)
        # no datastore stuff, purely using pre-loaded text
        responses = [self.global_rag.bm25_search(questionlist[i], idstr, k=k) for i in range(len(questionlist))]
        return responses

    def compose_all_bm25_reasoning(self, booklist, questionlist, idstr, words_per_chunk=500, k=1000):

        allresults = []
        for i in range(len(booklist)):
            allresults.append(self.bm25_reasoning(booklist[i], questionlist[i], k=1000))

        return [
            self.bm25_reasoning_all_easyindex(questionlist, idstr, k=k),
            allresults
        ]

    def easyindex_reasoning_all(self, booklist, questionlist, idstrs=None, words_per_chunk=500, k=1000, qbatchsize=2, singleindex=False):
        if idstrs is None:
            idstrs = [get_deterministic_hash(" ".join(booklist[i]) + self.embedding_model) for i in range(len(booklist))]
        elif isinstance(idstrs, str):
            idstrs = [idstrs]*len(booklist)
        passages = self.process_input(booklist, words_per_chunk=words_per_chunk)
        # breakpoint()
        plens = [len(p) for p in passages]
        pflat = [item for sublist in passages for item in sublist]
        pflat = [passage['text'] for passage in pflat]
        if not all(self.global_rag.index_exists(idstrs[i]) for i in range(len(idstrs))):
            # TODO optimize to avoid re-embedding ids multiple times, assert that ids go along with identical stuff
            # TODO how to do things?
            allembs = self.global_rag.embed_with_multi_gpu(pflat)
            groupembs, groupdocs = [], []
            curidx = 0
            for i in range(len(plens)):
                groupembs.append(allembs[curidx:curidx+plens[i]])
                groupdocs.append(pflat[curidx:curidx+plens[i]])
                curidx += plens[i]
            # breakpoint()
            for i in range(len(groupembs)):
                self.global_rag.index_documents(groupdocs[i], index_id=idstrs[i], pre_embeds=groupembs[i])
        # get all query embeddings in one go
        query_embeddings = self.global_rag.embed_with_multi_gpu(questionlist, qtype="query", batch_size=qbatchsize)
        # breakpoint()
        if singleindex:
            # we don't need to search different indices for each query
            responses = self.global_rag.search(query_embeddings, index_id=idstrs[0], k=k, doembed=False)
        else:
            responses = []
            for i in tqdm(range(len(query_embeddings))):
                response = self.global_rag.search(query_embeddings[i], index_id=idstrs[i], k=k, doembed=False)
                responses.append(response)
        return responses
    
    def easyindex_reasoning_composed(self, booklist, questionlist, idlist_setups, words_per_chunk=500, k=1000, singleindex=False):
        fullresponses = []
        for idls in idlist_setups:
            fullresponses.append(self.easyindex_reasoning_all(booklist, questionlist, idls, words_per_chunk=words_per_chunk, k=k, singleindex=singleindex))
        return fullresponses