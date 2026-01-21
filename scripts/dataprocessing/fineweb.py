from typing import Any, Generator


from functools import partial
from datasets import Dataset
from datasets import load_dataset

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

if __name__ == "__main__":
    # get 1M random chunks from fineweb
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="default",
        split="train",
        streaming=True
    )
    fwfilts = dataset.filter(lambda x: x['token_count']<500 and x['token_count']>200 and x['score'] > 3 and x['language']=='en')
    fwfilts = fwfilts.shuffle(seed=42)
    downfwdata = fwfilts.take(1_000_000)

    ds = Dataset.from_generator(partial[Generator[Any, Any, None]](gen_from_iterable_dataset, downfwdata), features=downfwdata.features)
    ds.save_to_disk("propercache/data/datastores/finewebgiant")

    breakpoint()