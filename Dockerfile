# FROM python:3.12-slim

# # Install system deps for llama-cpp-python (OpenBLAS, etc.)
# RUN apt-get update && apt-get install -y gcc g++ make cmake libopenblas-dev git && apt-get clean

# # Set env for building llama-cpp-python with OpenBLAS
# ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=OFF"
# ENV FORCE_CMAKE=1

# # Copy and install requirements
# COPY requirements.txt /opt/
# RUN pip install --no-cache-dir -r /opt/requirements.txt

# # Copy code
# COPY medical_bot_main_file.py /opt/
# COPY run_server.sh /opt/
# RUN chmod +x /opt/run_server.sh

# # SageMaker expects code in /opt/ml/code, but model in /opt/ml/model (mounted later)
# WORKDIR /opt
# EXPOSE 8080

# # Run the server
# CMD ["/opt/run_server.sh"]







# # For running on local
# # Use a Python base image with a specific version for consistency
# FROM python:3.12-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Install system dependencies needed for llama-cpp-python
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     libopenblas-dev \
#     libblas-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy the requirements file and install Python dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your application code and the model
# COPY medical_bot_main_file.py .
# COPY medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf .

# # Expose the port that the FastAPI application will listen on
# EXPOSE 8000

# # This is the command that will be executed when the container starts
# # It runs the main file directly with uvicorn.
# # The host 0.0.0.0 is crucial for local access.
# CMD ["uvicorn", "medical_bot_main_file:app", "--host", "127.0.0.1", "--port", "8000"]







# Use a Python base image with a slim version for a smaller footprint
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for llama-cpp-python and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file
COPY medical_bot_main_file.py .
COPY model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf ./model/

# Create logs directory for chat history
RUN mkdir -p logs

# Expose the port FastAPI will run on
EXPOSE 8000

# Set environment variables for Python and FastAPI
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the FastAPI application with uvicorn
CMD ["uvicorn", "medical_bot_main_file:app", "--host", "0.0.0.0", "--port", "8000"]