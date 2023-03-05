import sys
import pinecone
import openai
import os
import base64


class Cache:
    """ OpenAI Cache using Pinecone """
    __slots__ = 'openaiapi_key', 'pinecone_key', 'pinecone_index'

    MODEL = 'text-embedding-ada-002'
    INDEX = 'openaicache'

    def __init__(self, openaiapikey, pineconekey):
        self.openaiapi_key = openaiapikey
        self.pinecone_key = pineconekey
        openai.api_key = self.openaiapi_key
        pinecone.init(
            api_key=self.pinecone_key, environment="us-west1-gcp")
        self.pinecone_index = pinecone.Index(self.INDEX)

    def getcompletion(self, prompt):
        res = openai.Completion.create(engine='text-davinci-003', prompt=prompt, temperature=0,
                                       max_tokens=3000, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None)
        return res['choices'][0]['text'].strip()

    def __get_from_pinecone(self, embed):
        return self.pinecone_index.query(vector=embed, top_k=3, include_values=False, include_metadata=True)


    def __put_pinecone_with_completion(self):
        pass
    
    def getitem(self, data):
        # get an embedding from OpenAI as key
        embed = openai.Embedding.create(input=[data], model=self.MODEL)[
            'data'][0]['embedding']
        #print(embed)
        # cache lookup in PC
        pc_lookup = self.__get_from_pinecone(embed)
        #print(pc_lookup)
        if not pc_lookup.matches:
            vector_id = base64.b64encode(data.encode('utf-8'))
            completion = self.getcompletion(data)
            v = {'id':str(vector_id), 'values':embed, 'metadata':{'q':data, 'a':completion}}
            print(v)
            upsert_response = self.pinecone_index.upsert(vectors=[v])
            return completion
        
        first_match = pc_lookup.matches[0]
        score = first_match['score']
        print(score)
        if abs(score) < .5:
            return pc_lookup.matches[0]['metadata']['a']
        print("return nothong")
        return ''
