

from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TfidfRetriever
import os
from haystack.utils import print_answers
from haystack.nodes import FARMReader, TransformersReader
from haystack.document_stores import InMemoryDocumentStore

reader =  TransformersReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad")
pipe = None


dicts = []

def train(new_info):
    document_store = InMemoryDocumentStore()
    global dicts
    dicts.append({'content': new_info})
    document_store.write_documents(dicts)
    retriever = TfidfRetriever(document_store=document_store)
    global pipe
    pipe = ExtractiveQAPipeline(reader, retriever)

def ask(question):
    global pipe
    prediction = pipe.run(
        query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )
    return (find_answer(prediction))

def find_answer(prediction):
    answer_qualifications = {}
    answers = []
    for a in prediction['answers']:
        answer = a.answer
        for b in prediction['query'].split(" "):
            answer = answer.replace(b,"") 
        answers.append([answer, a.score])
    for i, a in enumerate(answers):
        answer_qualifications[i] = 0
        for b in answers:
            if (a[0] in b[0] or b[0] in a[0]) and a[0] != '':
                answer_qualifications[i] += a[1]
    highest_num = float(0)
    index_of_highest = None
    for a in answer_qualifications:
        if answer_qualifications[a] > highest_num:
            highest_num = answer_qualifications[a]
            index_of_highest = a
    return (prediction['answers'][index_of_highest].answer)




train("Skyliner is a boy, he likes beetles. He is 19 years old")
train("Conner is a boy, he likes angles, he is 21 years old")
print(ask("how old is Conner"))
