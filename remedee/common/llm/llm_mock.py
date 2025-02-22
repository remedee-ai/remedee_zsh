import os
from remedee.common.llm.llm import LLM

class LLMMock:
    def __init__(self, model=None):
        self.responses = []
        self.next_response = 0
        self.model = {"name": f"mock {model}"}

    async def query(self, messages, cache=None, stream_callback=None, temperature=None, background=False, prompt_name=None):
        if background:
            return "", 0
        response = self.responses[self.next_response]
        self.next_response += 1
        if stream_callback:
            await stream_callback(response, None)

        return response, 0
    
''' LLM that automatically saves responses to a given folder for testing
    purposes. This is useful for E2E tests where we want to cache the
    responses for a given input, but also want to regenerate the responses
    when the input changes. 
'''
class LLMWithMockedResponses(LLM):
    def __init__(self, model, cache_prefix, max_output_tokens=None, instance="default"):
        super().__init__(model, cache_prefix, max_output_tokens=max_output_tokens, instance=instance)
        self.cache = True

        # when we store the cache in the repo to fix a test, we want to delete the cache if the test fails
        # to regenerate it and be able to see the changes
        #
        # for tests that do not store the cache in the repo, the value can be set to False; however, it means 
        # the cache will grow indefinitely
        self.clear_cache_on_failure = True
        self.cache_size = len(self.request_cache)

    def _response_from_cache(self, messages):
        key, value = super()._response_from_cache(messages)

        # if there is a cache miss, and the initial size was not zero, delete the cache and throw an error
        if value is None and self.cache_size > 0 and self.clear_cache_on_failure:
            cache_file = self._get_cache_file()
            os.remove(cache_file)
            raise Exception("Response not in cache, cache file deleted! Rerun the test to regenerate the cache.")
        
        return key, value