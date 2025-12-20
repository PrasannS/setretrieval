# LIST OF PROMPTS USED FOR VARIOUS PURPOSES, NOTE THAT SEVERAL ARE USED WITH CACHING, SO CHANGING MAY LEAD TO RE-GENERATING CACHE

wholebook_prompt = """
You are an expert at analyzing text passages and answering complex yes/no questions that require deep understanding.

QUESTION: {}

BOOK (RAW TEXT): {}

QUESTION (RESTATED): {}

Answer with brief descriptions of relevant passages, and then at the end list all the passage IDs of the relevant passages. 

The format of the final answer should be: "Answer: Passage 1, Passage 80, Passage 100... (END)", or if there's no relevant passages, say "Answer: NO RELEVANT PASSAGES"
"""

aggregated_prompt = """
You are an expert at analyzing text passages and answering complex yes/no questions that require deep understanding.

Notes from LLM about different passages in book (ordered by passage number): {}

QUESTION: {}

First think step by step, and then at the end answer with 1-3 sentences based on analyzing the note. The format should be: [THINKING] ANSWER: [ANSWER]
"""

decomposed_prompt = """
You are an expert at analyzing text passages. You are given a passage from a book, and a query, and you need to do 1 of 2 things: 
1. If the passage may be relevant to the query, generate a brief (1-2 sentence) note specifying and explaining the specific relevant information from the passage.
2. If the passage is not relevant to the query, say "NOT RELEVANT". 
Don't generate any other text. Focus only on analyzing the given passage.

BOOK (RAW TEXT): {}
QUESTION: {}

Please answer with only the note, no other text.
"""

# unified to just 1 v2 prompt
decomposed_prompt_restrictive_4B = """You're given a question and a passage. Answer whether the passage directly applies to the question. Relevant questions might not always be obvious, and may require deeper reading, but if the passage doesn't directly apply to the question you should answer no.

QUESTION: {}

PASSAGE: {}

Answer whether the passage directly applies to the question. Answer with just yes or no, don't include any other text.
"""

decomposed_prompt_restrictive_8B = """
You are an expert at analyzing text passages. You are given a passage and a query used to retrieve documents. Based on reading the passage carefully, you need to determine if the passage meets the criteria of the query.
Passages that are not directly relevant to the query should be considered as not relevant, but different kinds of passages might be relevant.

QUERY: {}

PASSAGE (RAW TEXT): {}

Please answer with yes or no, no other text.
"""

decomposed_prompt_restrictive_oai = """
You are an expert at analyzing text passages. You are given a passage and a query used to retrieve documents. You need to determine if the passage meets the criteria of the query.
Passages that are not directly relevant to the query should be considered as not relevant.

QUERY: {}

PASSAGE (RAW TEXT): {}

Please answer with yes or no, no other text.
"""


categorize_prompt = """
You're given a question which is used to retrieve sets of passages from a diverse passage set(containing wikipedia, news, textbooks, novels, blogs, etc.). Categorize it into one of the following categories:
- Abstract: The question is about properties of passages that could apply to very different topics, subjects, or people. Example 1: Which passages describes a characteristic's evolution being influenced by the interaction between different groups? Example 2: What passage mentions an individual working temporarily with a foreign counterpart of their own organization? Example 3: What passage describes a series of investigations into a single subject over a long period of time?
- Semi-Specific: The rough topic (e.g. military, art, sports, modern fiction, etc.) is clear from the question, but it could still apply to different passages within the topic. Example 1: Which passages explains how a change in political leadership enabled the creation of an artistic work? Example 2: What passages describe a situation where the estimated number of casualties is given, but specific details about them are unknown?
- Specific: The question is about specific details in a passage, topic of passage is clear from question. Example: Which passages details a previously successful soccer team winning consecutive championships after more than a decode of losing?
- Stylistic: The question is related to formatting or structure of the passage, and not related to content of a passage. Example: Which passages contain multiple lists where the entries are defined by a title and a time period?

Question: {}

Please answer with one of the following categories: Abstract, Specific, Semi-Specific, Stylistic.
No other text.
"""

conceptual_rephrase_prompt = """
You're given a question. Rephrase it in a way that sounds more like a broader, more natural question that a person might ask, but captures the same question, and would be relevant to the same passages. 
Examples: 

Input 1: Which passage discusses recommendations for legal reform that explicitly address the needs of vulnerable populations? 
Output 1: What legal reform recommendations have explicitly addressed needs of vulnerable populations?

Input 2: What passages describe an object being acquired by one organization, but not used, before being acquired by another organization?
Output 2: What are things that, after being acquired by a first organization, went unused before being acquired by a second organization?

Input 3: Which passage describes a project as being a middle installment of a larger series?
Output 3: What middle projects are parts of bigger series?

Input 4: What passages describe a cyclical pattern of escalating retaliatory actions between opposing groups?
Output 4: What are examples of escalating retaliation between groups?

Input 5: What passage mentions an individual working temporarily with a foreign counterpart of their own organization?
Output 5: When have individuals temporarily worked with foreign counterparts in similar roles?

Input 6: Which passage describes a situation where an official statement is contradicted by information from professional sources, leading to repercussions for those who revealed the information?
Output 6: When have official and professional sources contradicted each other?

Question: {}
Please answer with the rephrased question. No other text.
"""

abstract_questions_prompt = """You are a helpful assistant that generates questions from a given text. Given a passage, generate a question about high-level structure or ideas in the passage, that can be used to retrieve the passage from a corpus of other ones, but will not be true for a related distractor passage.

Generate a list of 3 straightforward, unambiguous, unique, and high-level questions about properties of the passage. Avoid using any proper nouns, terms related to the passage, or questions that can be connected to passage content without deeper reasoning. 
Each question should clearly apply to the passage, but be abstract enough that the topic or category of the passage can't be identified by the question. It shouldn't be clear if the passage is about history, sports, a book passage, biology, etc.

EXAMPLE PASSAGE: {}

ABSTRACT QUESTIONS FOR EXAMPLE PASSAGE: {}

Now, generate questions based on this passage:

PASSAGE TO USE: {}

The questions should be diverse. Generate in a bulleted list with no other text."""

sciabstract_questions_prompt = """You are a helpful assistant that generates questions from a given text. Given a scientific abstract, generate a question about high-level concepts orr ideas in the passage, that can be used to retrieve the abstract from a corpus of other ones, but will only be true for a few other related abstracts.

Generate a list of 3 straightforward, unambiguous, unique, and high-level questions about properties of the passage. The question can mention relevant domain-specific ideas, but shouldn't be too specific to the abstract.
Each question should clearly apply to the passage, but be abstract enough that the content of the abstract can't be fully identified by the question. If possible ask questions that could be related to different scientific fields.

EXAMPLE ABSTRACT: {}

ABSTRACT QUESTIONS FOR EXAMPLE ABSTRACT: {}

Now, generate questions based on this abstract:

ABSTRACT TO USE: {}

The questions should be diverse. Generate in a bulleted list with no other text."""

example_abstract_passage = """In many species, only a small fraction of the total sequence of the genome encodes protein. For example, only about 1.5 percent of the human genome consists of protein-coding exons, with over 50 percent of human DNA consisting of non-coding repetitive sequences.[98] The reasons for the presence of so much noncoding DNA in eukaryotic genomes and the extraordinary differences in genome size, or C-value, among species, represent a long-standing puzzle known as the "C-value enigma".[99] However, some DNA sequences that do not code protein may still encode functional non-coding RNA molecules, which are involved in the regulation of gene expression.[100]"""

example_abstract_questions = "What passages discuss phenomena that are still not fully understood? What are cases where proportionally small groups can have disproportionate influence on a larger system?"
