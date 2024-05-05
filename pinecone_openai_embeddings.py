import os
from dotenv import load_dotenv
from datasets import load_dataset

from pinecone import Pinecone
import time
from pinecone import ServerlessSpec, PodSpec

from tqdm import tqdm  # progress bar
from openai import OpenAI

data = []
contexts = []
pinecone_api_key = ""
environment = ""

pc_client = None
pc_index = None

openAI_model = None
openAI_client = None
openAI_org = ""

sentences = [
    "the hive of bees protect their queen",  # 0
    "a beehive is an enclosed structure in which honey bees live",  # 1
    "a condominium is an enclosed structure in which people live",  # 2
    "the flying stinging insects guard the matriarch",  # 3
]


def init_data():
    # load in your dataset
    global data
    data = load_dataset("squad_v2", split="train")

    # specifically separate the 'context' value from each row SPECIFIC to the (squad_v2)  dataset
    global contexts
    contexts = list(set(data["context"]))


def init_pinecone():
    use_serverless = False

    # initialize connection to pinecone (get API key from .env file)
    global pinecone_api_key
    pinecone_api_key = os.environ["PINECONE_API_KEY"]

    # configure pinecone client
    global pc_client
    pc_client = Pinecone(api_key=pinecone_api_key)

    # define pinecone environment
    global environment
    environment = "gcp-starter"

    if use_serverless:
        spec = ServerlessSpec(cloud="aws", region="us-west-2")
    else:
        spec = PodSpec(environment=environment)

    index_name = "squad-search"

    # search all values under the key 'name' in the list of indexes under your pinecone account
    if index_name not in pc_client.list_indexes().names():
        pc_client.create_index(index_name, dimension=1536, metric="cosine", spec=spec)
        while not pc_client.describe_index(index_name).status["ready"]:
            time.sleep(2)

    # initialize pinecone index in a variable
    global pc_index
    pc_index = pc_client.Index(index_name)


# TODO make this more dynamic where it can take in a different type of data, particularily in batching too
def upsert_data(data):
    global openAI_client
    global openAI_org
    openAI_org = os.environ["OPENAI_ORG"]
    openAI_client = OpenAI(organization=openAI_org)

    global openAI_model
    openAI_model = "text-embedding-ada-002"

    batch_size = 100  # 4 or 100
    for i in tqdm(range(0, len(data), batch_size)):

        # find the end of the batch
        i_end = min(i + batch_size, len(data))

        # batch the contexts between the correct batch indices e.g. between 0-99, 100-199, 200-299
        batch = data[i:i_end]
        id_batch = [str(x) for x in range(i, i_end)]

        res = openAI_client.embeddings.create(model=openAI_model, input=batch)
        embeds = [r.embedding for r in res.data]

        metadata = [{"context": x} for x in batch]

        to_upsert = zip(id_batch, embeds, metadata)
        pc_index.upsert(vectors=to_upsert)


def format_pinecone_response(pinecone_response):
    formatted_response = []
    for match in pinecone_response["matches"]:
        context = match["metadata"]["context"]
        score = match["score"]
        formatted_response.append(f"[{round(score, 2)}]: {context}")
    return formatted_response


def test_query(query):
    global openAI_client
    global openAI_org
    openAI_org = os.environ["OPENAI_ORG"]
    openAI_client = OpenAI(organization=openAI_org)

    global openAI_model
    openAI_model = "text-embedding-ada-002"

    res = openAI_client.embeddings.create(input=[query], model=openAI_model)
    xq = res.data[0].embedding

    res = pc_index.query(vector=xq, top_k=3, include_metadata=True)

    for result in format_pinecone_response(res):
        print(result)


def main():
    load_dotenv()
    # init_data()
    init_pinecone()
    upsert_data(sentences)
    test_query("Can you tell me about bees and how they protect their queen?")


if __name__ == "__main__":
    main()
