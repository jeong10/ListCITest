FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run any script in the repo, e.g.:
#   docker run --rm listci main.py gmp graphs/paper/fig4b.txt
#   docker run --rm listci test_model.py graphs/sachs/graph_gt.txt datasets/sachs/dataset.csv
ENTRYPOINT ["python3"]
CMD ["main.py", "clmp", "graphs/paper/fig4b.txt"]
