

from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TfidfRetriever
import os
from haystack.utils import print_answers
from haystack.nodes import FARMReader, TransformersReader
from haystack.document_stores import InMemoryDocumentStore



reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                            tokenizer="distilbert-base-uncased", use_gpu=-1)

document_store = InMemoryDocumentStore()
dicts = [{'content': 'My name is Conner Frost.My age is 20 years old. I like drinking. I am a boy'}, {'content': 'Robert Downy is Iron man'}]
document_store.write_documents(dicts)
retriever = TfidfRetriever(document_store=document_store)
pipe = ExtractiveQAPipeline(reader, retriever)


prediction = pipe.run(
    query="Who is Iron man", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
print(prediction['answers'][0].answer)

