

from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TfidfRetriever
import os
from haystack.utils import print_answers
from haystack.nodes import TransformersReader
from haystack.document_stores import InMemoryDocumentStore


document_store = InMemoryDocumentStore()


dicts = [{'content': 'My name is Ahmed Shahriar. I am 19 years old. I like programming. I am a boy'}]
document_store.write_documents(dicts)
retriever = TfidfRetriever(document_store=document_store)

reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                            tokenizer="distilbert-base-uncased", use_gpu=-1)

pipe = ExtractiveQAPipeline(reader, retriever)
prediction = pipe.run(
    query="What is my name", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)


print_answers(prediction, details="mininal")
