# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the app's port
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]
