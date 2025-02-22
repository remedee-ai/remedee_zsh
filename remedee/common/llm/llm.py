import asyncio
from datetime import datetime
import json
import logging
import os
from openai import APIConnectionError, APIError, AsyncOpenAI
from openai import RateLimitError
from openai import AsyncAzureOpenAI
import backoff
import aiofiles

from remedee.common.version import product_version_long
from remedee.common.utils.keychain import get_env

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import tiktoken
gpt_35_encoding = tiktoken.get_encoding("cl100k_base")
request_cache = {}
full_log_id = 0

def count_tokens(string: str) -> int:
    return len(gpt_35_encoding.encode(string))

def load_request_cache(cache_file):
    global request_cache
    if cache_file in request_cache:
        return request_cache[cache_file]
    
    cache = {}
    try:
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                cache[entry[0]] = entry[1]
    except FileNotFoundError:
        pass

    request_cache[cache_file] = cache
    return cache

def append_cache(cache_file, key, value):
    global request_cache
    if cache_file not in request_cache:
        request_cache[cache_file] = {}
    
    cache = request_cache[cache_file]
    if key in cache:
        return
    
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    cache[key] = value

    # save the cache
    with open(cache_file, "a") as f:
        f.write(json.dumps([key, value]) + "\n")

instances = set()

def create_client(model_id, model):
    azure_api_key = None
    if model_id == "4o":
        # self-payed
        #azure_endpoint = "https://remedee.openai.azure.com/" #os.environ["AZURE_OPENAI_ENDPOINT"]
        #azure_api_key = get_env("AZURE_OPENAI_API_KEY", 'azure-open-ai-api', 'remedee')
        
        # MS Founders
        # NOTE: remedee-ms (which is central europe) does not have the latest 4o, that's why we use swe
        # azure_endpoint = "https://remedee-ms.openai.azure.com/" #os.environ["AZURE_OPENAI_ENDPOINT"]
        # azure_api_key = get_env("AZURE_OPENAI_API_KEY", 'azure-open-ai-api-ms', 'remedee')
        #
        azure_endpoint = "https://remedee-swe.openai.azure.com/" #os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_api_key = get_env("AZURE_OPENAI_API_KEY", 'azure-open-ai-api-swe', 'remedee')
    elif model_id == "4o-mini" or model_id == "4":
        # MS Founders
        azure_endpoint = "https://remedee-eastus.openai.azure.com/"
        azure_api_key = get_env("AZURE_OPENAI_API_KEY_US", 'azure-open-ai-api-eastus', 'remedee')
    else:
        print(f"WARNING: No Azure API key found for model {model_id}, using OpenAI API")
        return AsyncOpenAI(api_key=get_env('OPENAI_API_KEY', 'open-ai-api', 'asr'),)
    
    if azure_api_key is None:
        raise Exception("Azure API key not found")
    
    if len(azure_api_key) < 10:
        raise Exception(f"Azure API key too short: '{azure_api_key}'")
    
    api_key_hash = int(hash(azure_api_key)) % 10000
    #print(f"-- Using Azure API key with hash {api_key_hash} for model {model_id}")
    #print(f"-- Azure API key: '{azure_api_key}'")
    
    azure_deployment = model["name"]
    azure_api_version = "2024-02-15-preview"

    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=azure_api_key,
        api_version=azure_api_version
    )
    
class LLMLogger:
    def __init__(self, log_file_name, file_header=None):
        self.log_file = f".log/{log_file_name}"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.file_header = file_header
        self.lock = asyncio.Lock()

    async def log(self, data, additional_log_path = None):
        async with self.lock:  # Ensure that writes are thread-safe
            # check if file exists and write header if not
            if self.file_header and not os.path.exists(self.log_file):
                data = self.file_header + data

            # if there is an additional data log, log long messages only there and shorten them for the main log
            if additional_log_path:
                async with aiofiles.open(additional_log_path, 'a') as file:
                    await file.write(data)
                if len(data) > 1000 and not data.endswith("\033[0m"):
                    data = data[:1000] + "...\n\n"

            async with aiofiles.open(self.log_file, 'a') as file:
                await file.write(data)

class TokenRate:
    def __init__(self) -> None:
        self.sum = 0
        self.sum_minute = 0
        self.current_minute = datetime.now().minute
        self.last_log = datetime.now()

    def add(self, current_minute, tokens):
        ret_minute_sum = None
        ret_minute_rate = None
        if current_minute != self.current_minute:
            ret_minute_sum = self.sum_minute
            ret_minute_rate = ret_minute_sum / (datetime.now() - self.last_log).total_seconds()
            self.last_log = datetime.now()

            self.current_minute = current_minute
            self.sum_minute = 0

        self.sum += tokens
        self.sum_minute += tokens
        return ret_minute_sum, ret_minute_rate


class LLM:
    instance_cheap = None
    instance_validation = None

    def __init__(self, model=None, cache_prefix=".cache/request_cache", cache=False, max_output_tokens = None, instance="default"):
        if model is None:
            model = "4o"

        if max_output_tokens is None:
            max_output_tokens = 1500
        elif max_output_tokens > 4096:
            raise Exception("max_output_tokens must be <= 4096")

        self.verbose = True
        self.max_output_tokens = max_output_tokens
        self.models = {
            "3.5": {
                "name": "gpt-3.5-turbo",    # gpt-3.5-turbo-1106
                "max_tokens": 16385,
            },
            "4o": {
                "name": "gpt-4o",
                "max_tokens": 128000,
            },
            "4o-mini": {
                "name": "gpt-4o-mini",
                "max_tokens": 128000,
            },
            "4": {
                "name": "gpt-4-turbo", # "gpt-4-turbo-preview", # gpt-4-1106-preview
                "max_tokens": 128000,
            }
        }

        # the model spec tokens, are max context length including output as it is auto regressive -> compute max_input_tokens
        for key in self.models:
            self.models[key]["max_input_tokens"] = self.models[key]["max_tokens"] - self.max_output_tokens
        
        self.model_key = f"{model:.10g}" if isinstance(model, float) else str(model)
        self.model = self.models[self.model_key]
        self.fallback_model = self.models["4o-mini"] if self.model_key != "4o-mini" else None
        self.temperature = 0.2
        self.client = create_client(self.model_key, self.model)
        self.client_provider = self.client.user_agent.split("/")[0] + "/" + self.client.base_url.host.split(".")[0]
        self.cache = cache
        self.cache_prefix = cache_prefix
        self.request_cache = load_request_cache(self._get_cache_file())
        self.message_token_overhead = count_tokens("role: user") + count_tokens("content:")

        # must have different instances, which can not run in parallel and log to the same file
        if instance in instances:
            raise Exception(f"Instance {instance} already exists")
        instances.add(instance)

        log_suffix = f"_{instance}" if instance != "default" else ""
        self.forground_log = LLMLogger(f"llm{log_suffix}.log")
        self.backround_log = LLMLogger(f"llm{log_suffix}_background.log")
        self.stats_log = LLMLogger(f"llm_stats_{self.model_key}.csv", "t,minute_in,rate_in,minute_out,rate_out,total_in,total_out\n")
        self.input_tokens = TokenRate()
        self.output_tokens = TokenRate()
        
    
    def next_log_path(self):
        global full_log_id
        full_log_id = (full_log_id + 1) % 100

        # create log path of form .log/full_log/001.log
        path = f".log/full_log/{full_log_id:03d}.log"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)

        return path

    async def log(self, data, backround, full_log_path=None):
        logger = self.backround_log if backround else self.forground_log
        await logger.log(data, full_log_path)

    async def log_stats(self, input, output, backround):
        # save total before updating
        in_total = self.input_tokens.sum
        out_total = self.output_tokens.sum

        t = datetime.now()
        current_minute = t.minute
        minute_in, rate_in = self.input_tokens.add(current_minute, input)
        minute_out, rate_out = self.output_tokens.add(current_minute, output)
        
        if minute_in is not None:
            t = t.strftime("%Y-%m-%d %H:%M:%S")
            await self.stats_log.log(f"{t},{minute_in},{rate_in:.2f},{minute_out},{rate_out:.2f},{in_total},{out_total}\n")

    def _get_cache_file(self):
        return f"{self.cache_prefix}_{self.model['name']}.txt"
    
    def get_messages(self, query, sys_prompt = None):
        query_message = {
                "role": "user",
                "content": query
            }
        
        if not sys_prompt:
            return [query_message]
        
        system_message = {
            "role": "system",
            "content": sys_prompt
        }
        return [system_message, query_message]
    
    # handle 429 and connection issues
    @backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError), max_time=10)
    async def _start_completion(self, model, messages, temperature=None):
        response = await self.client.chat.completions.create(
            model=model["name"],
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_output_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True,
            # stop=["\n"]
        )
        return response
    
    async def _select_model(self, input_tokens, background):
        model = self.model
        await self.log(f"Model: {model['name']} ({self.client_provider})\n", background)

        # we use a buffer here to be on the safe side and not have issues due to incorrect computation of tokens in rare cases
        if input_tokens + 10 > model["max_input_tokens"]:
            if self.fallback_model:
                self.log(f"WARNING: Input tokens of {input_tokens} exceeds the maximum supported by model {model['name']} of {model['max_input_tokens']}, falling back to {self.fallback_model['name']}")
                model = self.fallback_model
                await self.log(f"Fallback to model: {model['name']} ({input_tokens} tokens exceeds the maximum supported by model)\n", background)
        if input_tokens > model["max_input_tokens"]:
            raise Exception(f"Input tokens of {input_tokens} exceeds the maximum supported by model {model['name']} of {model['max_input_tokens']}")
        
        return model
    
    async def _stream_response(self, response, background, full_log_path, stream_callback=None, log_tokens=True):
        role = None
        can_proceed = True
        response_text = ""
        event = None

        async for event in response:
            if len(event.choices) == 0:
                # pause shortly to avoid busy waiting
                await asyncio.sleep(0.01)
                continue

            event_text = event.choices[0].delta
            if event_text.role is not None:
                role = event_text.role
            token = event_text.content

            if token is None or token == "":
                await asyncio.sleep(0.01)
                continue

            response_text += token

            if stream_callback:
                can_proceed = await stream_callback(response_text, token)

            if log_tokens:
                await self.log(token, background, full_log_path)

            if not can_proceed:
                break

        if event is not None and event.choices[0].finish_reason is not None:
            finish_reason = event.choices[0].finish_reason
            if finish_reason != "stop":
                await self.log(f"Finish reason: {finish_reason} ({extra})\n", background)
            if finish_reason == "content_filter":
                extra = event.choices[0].model_extra
                raise Exception(f"Finish reason: {finish_reason} ({extra})")

        return response_text

    async def _execute_query_retry(self, messages, input_tokens, background, full_log_path, stream_callback=None, temperature=None, log_response=True):
        model = await self._select_model(input_tokens, background)
        try:
            response = await self._start_completion(model, messages, temperature)
        except APIError as e:
            # We do exactly one retry as the initiation of the request can fail.
            # Unfortunately, we don't know the exact reason for the failures that we saw.
            # NOTE: 
            # - rate limit errors are handled separately by the backoff decorator
            # - we do not retry in case of content_filter errors
            await self.log(f"APIError: {e}\n", background)
            if e.code and 'content_filter' in e.code:
                raise Exception(f"Query failed: content_filter ({e.message})")
            
            # retry; exception are handle further up
            response = await self._start_completion(model, messages, temperature)
        
        # the streaming process may fail too, but we will not catch any errors here and let them bubble up
        return await self._stream_response(response, background, full_log_path, stream_callback, log_tokens=log_response)
    
    def _append_cache(self, key, value):
        append_cache(self._get_cache_file(), key, value)

    def _response_from_cache(self, messages):
        # TODO: race condition
        key = " ".join([message["content"] for message in messages])
        value = self.request_cache[key] if key in self.request_cache else None
        return key, value

    async def query(self, messages, cache=None, stream_callback=None, temperature=None, background=False, prompt_name=None):
        use_cache = cache if cache is not None else self.cache
        cache_key, response_text = self._response_from_cache(messages) if use_cache else (None, None)
        full_log_path = self.next_log_path()
            
        def to_gpt_message(message):
            role = message["role"]
            current_time = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

            # don't add timestamps to assistent messages, otherwise they may respond with timestamps
            t = message["t"] + ": " if role == "user" and "t" in message else ""
            return {
                "role": role,
                "content": t + message["content"]
                    .replace("{current_time}", current_time)
                    .replace("{release_version}", f"{product_version_long}")
            }
        # remove timestamps from each message and instead add it to "content"
        messages = [to_gpt_message(message) for message in messages]

        context_tokens = 0
        for message in messages[1:-1]:
            context_tokens += count_tokens(message["content"])
            context_tokens += self.message_token_overhead

        prompt_tokens = count_tokens(messages[0]["content"]) + self.message_token_overhead
        query_tokens = count_tokens(messages[-1]["content"]) + self.message_token_overhead
        input_tokens = prompt_tokens + context_tokens + query_tokens + 1

        prompt_name_log = f" ({prompt_name})" if prompt_name else ""
        await self.log(f"PROMPT  ({prompt_tokens}){prompt_name_log}: {messages[0]['content']}\n\n", background, full_log_path)
        if context_tokens > 0:
            await self.log(f"CONTEXT ({context_tokens}): {messages[1]['content']}\n\n", background, full_log_path)
        else:
            await self.log(f"CONTEXT (0): NONE\n\n", background, full_log_path)
        await self.log(f"\033[1m\033[37mQUERY   ({query_tokens}): {messages[-1]['content']}\n\n\033[0m", background, full_log_path)

        t_start = datetime.now()
        # return the cache only after we have the input tokens
        if response_text is None:
            await self.log(f"\033[33mRESPONSE: ", background, full_log_path)

            # log while streaming is useful to understand the progress while debugging or when a failure occurs
            # however, parallel requests will mix the logs
            log_stream = False
            response_text = await self._execute_query_retry(messages, 
                                                            input_tokens, 
                                                            background, 
                                                            full_log_path, 
                                                            stream_callback, 
                                                            temperature,
                                                            log_response=log_stream)   # _execute_query logs the response token by token
            if not log_stream:
                await self.log(response_text, background, full_log_path)

            output_tokens = count_tokens(response_text)
            if use_cache:
                self._append_cache(cache_key, response_text)
            await self.log("\033[0m", background, full_log_path)
            await self.log_stats(input_tokens, output_tokens, background)
        else:
            await self.log(f"\033[33mCACHED_RESPONSE: " + response_text + "\033[0m", background, full_log_path)

        t = (datetime.now() - t_start).total_seconds()

        if self.verbose:
            response_short = response_text[:40].replace("\n", " ")
            logging.info(f"Response: {response_short}...")
        
        time_log = f" m={self.model_key} ({self.client_provider}), p={prompt_name if prompt_name else ''}, input_length={input_tokens}, t={t:.3f}s"
        n = 116 - len(time_log)
        fill_line = " " * n
        format = "\033[1m\033[37m\033[100m"
        #format = "\033[7m"
        await self.log(f"\n\n{format}{time_log}{fill_line}\033[0m", background, full_log_path)
        #await self.log(f"{end_of_log}\n\n", background)

        if stream_callback:
            await stream_callback(response_text, None)
        return response_text, input_tokens

    async def query_simple(self, query):
        messages = self.get_messages(query)
        return await self.query(messages)

LLM.instance_cheap = LLM(model="4o-mini", cache=True, instance="cheap")
LLM.instance_cheap.verbose = False
LLM.instance_validation = LLM(model="4o", cache=True, instance="validation")
