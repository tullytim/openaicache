import sys
import pinecone
import openai
import os
import base64
from termcolor import colored

class Cache:
    """ OpenAI Cache using Pinecone """
    __slots__ = 'openaiapi_key', 'pinecone_key', 'pinecone_index', 'epsilon', 'model', 'pinecone_index_name'

    def __init__(self, openaiapikey, pineconekey, pinecone_index_name, epsilon=0.90, model='text-embedding-ada-002'):
        """ Epsilon is essentially cache aggressiveness """
        self.openaiapi_key = openaiapikey
        self.pinecone_key = pineconekey
        openai.api_key = self.openaiapi_key
        pinecone.init(api_key=self.pinecone_key, environment="us-west1-gcp")
        self.pinecone_index = pinecone.Index(pinecone_index_name)
        self.epsilon = epsilon
        self.model = model
        self.pinecone_index_name = pinecone_index_name

    def getcompletion(self, prompt):
        res = openai.Completion.create(engine='text-davinci-003', prompt=prompt, temperature=0,
                                       max_tokens=3000, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None)
        return res['choices'][0]['text'].strip()        

    def __get_from_pinecone(self, embed):
        return self.pinecone_index.query(vector=embed, top_k=3, include_values=False, include_metadata=True)

    def __put_pinecone_with_completion(self, embed, data, completion):
        vector_id = base64.b64encode(data.encode('utf-8'))
        v = {'id':str(vector_id), 'values':embed, 'metadata':{'q':data, 'a':completion}}
        self.pinecone_index.upsert(vectors=[v])
    
    def getitem(self, data):
        data = f"Answer the question below.\n\n\nQuestion: {data}\n\nAnswer:"
        embed = openai.Embedding.create(input=[data], model=self.model)['data'][0]['embedding']
        # cache lookup in PC
        pc_lookup = self.__get_from_pinecone(embed)
        if not pc_lookup.matches:
            print(colored('CACHE MISS', 'red', attrs=['bold']))
            completion = self.getcompletion(data)
            self.__put_pinecone_with_completion(embed, data, completion)
            return completion
        
        first_match = pc_lookup.matches[0]
        score = first_match['score']
        print(f"PINECONE SCORE: {score}")
        print(colored('Hello', 'black', attrs=['bold']))

        # Also a HIT
        if abs(score) > self.epsilon:
            print(colored('CACHE HIT', 'green', attrs=['bold']))
            return pc_lookup.matches[0]['metadata']['a']
        else:
            print(colored('CACHE MISS', 'red', attrs=['bold']))
            completion = self.getcompletion(data)
            self.__put_pinecone_with_completion(embed, data, completion)
            return completion
