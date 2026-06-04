FROM public.ecr.aws/lambda/python:3.12

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy source code
COPY src/ ./src/
COPY ml/ ./ml/

# Default handler (overridden per Lambda function)
CMD ["src.inference.handler.lambda_handler"]