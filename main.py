import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from typing import Optional, List, Mapping, Any

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage
    )
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    )
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings


from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

ckpt="Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto", trust_remote_code=True, bf16=True).eval()
#model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(ckpt, trust_remote_code=True)
#model = model.cuda()


class OurLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "qwen7bchat"
    dummy_response: str = "我的回复"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)
        text,_ = model.chat(tokenizer, prompt, history=[])
        return CompletionResponse(text=text)
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM()

# define embed model
Settings.embed_model = "local:bge-small-zh-v1.5"

 
# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./doc").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

 # Query and print response
query_engine = index.as_query_engine()

qs = ["发动机怎么造？",
      "详细说明起落架怎么设计，给出参考文献？", 
      "支撑结构的图片",
      "机身材料用什么，怎么制造？",
      "论文发表在哪里？期刊名字是什么"]

for q in qs:
    print(q+"============================")
    response = query_engine.query(q)
    print(response)


