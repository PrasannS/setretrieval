from setretrieval.datagen.generate_setdata import passages_to_questions
import argparse
from datasets import Dataset
from setretrieval.utils.utils import pickdump
from setretrieval.utils.constants import abstract_questions_prompt, sciabstract_questions_prompt, example_abstract_passage, example_abstract_questions

example_gut_passage = """'But never a word had Mr. Paul said about raising Tom\'s salary. Tom supposed he did not intend to raise it. And, much as he liked his post, and, for many reasons, his stay at Islip, he entertained notions of quitting both. Valentine had stopped the income his father had paid to Mrs. Chandler; and Tom\'s two hundred a-year, combined with the trifle remaining to her out of her private income, only just sufficed to keep the home going.It chanced that on the very same Sunday evening, when they were talking at North Villa of Valentine\'s doings, Tom broached the subject to his mother. They were sitting out of doors in the warm summer twilight, sniffing the haycocks in the neighbouring field. Tom spoke abruptly."I am thinking of it. You see, mother mine, there is no prospect of advancement where I am. It seems to me that I may jog on for ever at two hundred a-year----""As things are, yes: but nothing more. If--for instance--if I wanted to set up a home of my own, I have no means of doing it. Never shall have, at the present rate.""No. It is of no use to think of it. If I thought of it ever so, I could not do it. Putting that idea aside, it occurs to me sometimes to remember that I am eight-and-twenty, and ought to be doing better for myself."Opening the Bible on her lap, Mrs. Chandler took out the spectacles that lay between the leaves, and put them into their case with trembling fingers.'"""
example_gut_questions = "What are passages describing ideas or decisions prompted by financial constraints? What are passages that include descriptions of natural surroundings?"

example_sciabs_passage = """Title: Connecting Vision and Language with Localized Narratives Abstract: We propose Localized Narratives, a new form of multimodal image annotations connecting vision and language. We ask annotators to describe an image with their voice while simultaneously hovering their mouse over the region they are describing. Since the voice and the mouse pointer are synchronized, we can localize every single word in the description. This dense visual grounding takes the form of a mouse trace segment per word and is unique to our data. We annotated 849k images with Localized Narratives: the whole COCO, Flickr30k, and ADE20K datasets, and 671k images of Open Images, all of which we make publicly available. We provide an extensive analysis of these annotations showing they are diverse, accurate, and efficient to produce. We also demonstrate their utility on the application of controlled image captioning."""
example_sciabs_questions = "What are passages that describe non-standard ways of collecting human annotation data? What are passages that describe large-scale projects to collect common research resources?"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # propercache/data/datastores/gutenberg_train_240k_shuffled
    parser.add_argument("--dataset_path", type=str, default="propercache/data/datastores/fullabstractset10k_heldout")
    parser.add_argument("--startindex", type=int, default=0)
    parser.add_argument("--endindex", type=int, default=10000)
    parser.add_argument("--domain", type=str, default="wikipedia")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--doshuffle", type=str, default="no")

    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.dataset_path).select(range(args.startindex, args.endindex))
    if args.doshuffle == "yes":
        dataset = dataset.shuffle(seed=42)
        
    if args.domain == "wikipedia":
        pfunct = lambda x: abstract_questions_prompt.format(example_abstract_passage, example_abstract_questions, x)
    elif args.domain == "scientific":
        pfunct = lambda x: sciabstract_questions_prompt.format(example_sciabs_passage, example_sciabs_questions, x)
    elif args.domain == "gutenberg":
        pfunct = lambda x: abstract_questions_prompt.format(example_gut_passage, example_gut_questions, x)
    resultdata = passages_to_questions(dataset, model=args.model, pfunct=pfunct)

    resultdata.save_to_disk(f"{args.dataset_path}_{args.startindex}_{args.endindex}_{args.model}_shuff{args.doshuffle}_questions")

    breakpoint()



