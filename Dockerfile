FROM public.ecr.aws/lambda/python:3.11

# Copy shared ml modules as a Lambda layer under /opt/python
COPY ml/features.py      /opt/python/features.py
COPY ml/drift.py         /opt/python/drift.py

# Copy all Lambda handlers
COPY src/inference/handler.py    ${LAMBDA_TASK_ROOT}/inference/handler.py
COPY src/monitoring/handler.py   ${LAMBDA_TASK_ROOT}/monitoring/handler.py
COPY src/narration/handler.py    ${LAMBDA_TASK_ROOT}/narration/handler.py
COPY src/validation/handler.py   ${LAMBDA_TASK_ROOT}/validation/handler.py
COPY src/promotion/handler.py    ${LAMBDA_TASK_ROOT}/promotion/handler.py

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Default handler — overridden per Lambda on deploy
CMD ["inference.handler.lambda_handler"]