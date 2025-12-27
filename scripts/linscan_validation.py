# given questions, get linear scan labels with gemini first, then see if we can recover them with
# cheaper pipelines (e.g. using reranker, smaller api model, etc.)
import argparse
import os
from datasets import Dataset
from setretrieval.utils.utils import pickload
from tqdm import tqdm
from setretrieval.datagen.generate_setdata import chunks_to_inds
from setretrieval.inference.oai_request_client import ParallelResponsesClient
from setretrieval.datagen.generate_setdata import hierarchical_positive_search
from setretrieval.utils.utils import get_deterministic_hash


gutenuseqs = [
    # 'What are passages that explain the fundamental principles behind a technical activity?',
    'What are passages that illustrate a concept by first stating a general rule and then providing the consequences of deviating from it?',
    'Which passages offer guidance on personal conduct and achieving success?',
    "Which passages describe a character's exploration leading to the discovery of a vital resource?",
    "Which passages describe a situation involving a case of mistaken identity?"
]
wikiuseqs = [
    'Which passage describes a project being revived after a period of political suppression?',
    'Which passage describes a project as being a middle installment of a larger series?',
    'What passage describes a series of investigations into a single subject over a long period of time?',
    'What passage describes a subject for which a key piece of information remains unknown due to technical limitations?',
    "Which passage describes how a group's successful smaller operations contributed to an increase in its membership?"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qindex", type=int, default=0)
    parser.add_argument("--qset", type=str, default="guten")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--mkey", type=str, default="geminipro")
    args = parser.parse_args()

    # get gold labels with gemini-2.5-pro
    oai_client = ParallelResponsesClient(max_concurrent=100)

    qsets = {'guten': gutenuseqs, 'wiki': wikiuseqs}
    qset = qsets[args.qset]

    dpath = f"propercache/data/datastores/sanitychecks/{args.qset}sameidlists_{get_deterministic_hash(qset[args.qindex])}"
    passages = Dataset.load_from_disk(dpath)

    mkey=args.mkey
    amodel=args.model

    text = [row['text']+" " for row in passages]
    # difference between v1, v2 is 1 word change in the prompt (basically a seed consistency checker)
    gold_labels = hierarchical_positive_search(text, [qset[args.qindex]], get_deterministic_hash(dpath)+mkey+"v8", models=[amodel])

    # text = [row['text'] for row in passages]
    # # difference between v1, v2 is 1 word change in the prompt (basically a seed consistency checker)
    # gold_labels = hierarchical_positive_search(text, [qset[args.qindex]], get_deterministic_hash(dpath)+mkey+"v9", models=[amodel])

    # gold_labels = hierarchical_positive_search(passages['text'], [qset[args.qindex]], get_deterministic_hash(dpath)+"geminiflashv1", models=["gemini-2.5-flash-lite"])
    # breakpoint()

    # 6 and 7 are with new fs prompt (pos + neg)
    # 5 is with intermediate prompt (pos only)