import os
import json
import boto3
import pinecone
from tqdm import tqdm
from pinecone import Pinecone


def get_bedrock_embedding(text):
    """Generate embeddings using AWS Bedrock."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    body = {"inputText": text}
    response = client.invoke_model(
        body=json.dumps(body),
        modelId=os.getenv('EMBEDDING_MODEL_ID')
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def upload_to_pinecone(index, file_path):
    """Read the file, generate embeddings, and upload to Pinecone."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    embedding = get_bedrock_embedding(text)
    index.upsert(vectors=[("doc-" + os.path.basename(file_path), embedding, {"text": text})])


def main(folder_path):
    """Main function to process all files in the folder and upload embeddings."""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), pool_threads=30)
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for file_name in tqdm(files, desc="Uploading to Pinecone"):
        file_path = os.path.join(folder_path, file_name)
        upload_to_pinecone(index, file_path)

    print("All files processed successfully!")


if __name__ == "__main__":
    folder_path = "./documents"
    main(folder_path)
