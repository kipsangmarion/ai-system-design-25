FROM python:3.9-slim

# We need to upgrade Python 3 to at least 3.7 because pkgconf has format strings! yuck
# RUN apt-get update -y && \
#    apt-get install -y python3-pip python3-dev

WORKDIR /app

# We copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Environment variables
ENV FLASK_APP=base_iris_flask.py
ENV FLASK_RUN_PORT=4000
ENV FLASK_ENV=development

# Expose flask port
EXPOSE 4000

# Run the app
CMD [ "python", "base_iris_flask.py", "4000" ]

