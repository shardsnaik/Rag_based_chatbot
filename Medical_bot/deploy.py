# # deploy.py
# import boto3
# import sagemaker
# from sagemaker.pytorch import PyTorchModel
# from sagemaker.predictor import Predictor
# import time

# # Initialize session
# sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()  # Ensure proper IAM role

# # Upload model to S3
# model_data = sagemaker_session.upload_data(
#     path="model.tar.gz",
#     key_prefix="medical-bot/model"
# )

# # Create model
# pytorch_model = PyTorchModel(
#     model_data=model_data,
#     role=role,
#     framework_version='2.0.0',
#     py_version='py310',
#     entry_point='serve.py',
#     source_dir='.',
#     predictor_cls=Predictor
# )

# # Deploy endpoint
# predictor = pytorch_model.deploy(
#     initial_instance_count=1,
#     instance_type='ml.m5.xlarge',  # Adjust based on your needs
#     endpoint_name='medical-bot-endpoint'
# )

# print("Endpoint deployed:", predictor.endpoint_name)

# import requests

# API_URL = "https://h561hdi5s1r2d2tz.us-east-1.aws.endpoints.huggingface.cloud"
# headers = {
#     "Authorization": "Bearer os.getenv('hugging_face_api)",
#     "Content-Type": "application/json"
# }

# data = {
#     "inputs": "Your input text or data here"
# }

# response = requests.post(API_URL, headers=headers, json=data)
# print(response.json())
# pip install openai 

from openai import OpenAI 

client = OpenAI(
	base_url = "https://h561hdi5s1r2d2tz.us-east-1.aws.endpoints.huggingface.cloud/v1/",
	api_key = "os.getenv('hugging_face_api)"
)

chat_completion = client.chat.completions.create(
	model = "sharadsnaik/medgemma-4b-it-medical-gguf",
	messages = [
		{
            "role": "user",
            "content": "Describe the symptoms of dengue fever."
        }
	],
	stream = True
)

for message in chat_completion:
	print(message.choices[0].delta.content, end = "")
	#####################################################################