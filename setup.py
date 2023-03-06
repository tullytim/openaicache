from setuptools import setup

setup(
    name='openai-pinecone-cache',
    description='Library to cache openai data in Pinecone, helping to save cost but also provides fuzzy matching to cache inexact matches on prompts',
    version='0.1.0',
    url='https://github.com/tullytim/openaicache',
    author='Tim Tully',
    author_email='tim@menlovc.com',
    license='MIT',
    keywords='pinecone vector vectors embeddings database transformers openai cache',
    python_requires='>=3',
    py_modules=['pinecli'],
    install_requires=[
        'pinecone-client', 'pinecone-client[grpc]', 'openai', 'python-dotenv'
    ],
    entry_points={
    },
)
