import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Storage:
    def __init__(self):
        self.local_storage_path = "storage"
        os.makedirs(self.local_storage_path, exist_ok=True)

        # Initialize AWS S3 client if credentials are provided
        self.use_s3 = os.getenv("AWS_ACCESS_KEY") and os.getenv("AWS_SECRET_KEY")
        if self.use_s3:
            print("Initializing AWS S3 client...")
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            )
            self.bucket_name = os.getenv("AWS_BUCKET_NAME")
            print(f"S3 client initialized. Bucket: {self.bucket_name}")
        else:
            print("AWS S3 credentials not found. Using local storage.")

    def save_file(self, file_name: str, file_content: bytes) -> str:
        """
        Save a file locally or to S3 based on the environment.
        """
        if self.use_s3:
            print(f"Uploading file to S3: {file_name}")
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=file_name,
                    Body=file_content,
                )
                file_url = f"https://{self.bucket_name}.s3.amazonaws.com/{file_name}"
                print(f"File uploaded to S3. URL: {file_url}")
                return file_url
            except Exception as e:
                print(f"Error uploading file to S3: {e}")
                raise
        else:
            print(f"Saving file locally: {file_name}")
            file_path = os.path.join(self.local_storage_path, file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)
            return file_path