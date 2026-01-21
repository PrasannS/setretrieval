
# Repository for set retrieval project

### Setting up a fresh project (should come in handy for cluster use-cases as well)

1. Make sure to use a conda virtual environment (python==3.10) 
2. Make sure to set keys for: OPENAI_API_KEY, GEMINI_API_KEY (I recommend putting in bashrc)
3. Make sure to get wandb setup
3.5. Can install uv with `pip install uv`
3.7. Install pylate from latest github commit (to deal with larger vectors)
4. Run `uv pip install -e .`
5. TODO -> Any tests to run?

You should now be ready to go for all functionality!

### Code cleanliness TODOs
- Try to set up some actual tests
- Figure out the right requirements to reproduce different things (different envs may be required for stuff)
- TODO is datastore_reasoning really necessary?
