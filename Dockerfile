# Use official Python 3.12 image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip, wheel, setuptools

# Upgrade pip, wheel, setuptools
RUN pip install --upgrade pip wheel setuptools

# Install scientific packages first to ensure wheels are used
RUN pip install --prefer-binary numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1

# Copy requirements.txt and remove numpy, scipy, scikit-learn lines for the next step
RUN grep -v -i -E "^(numpy|scipy|scikit-learn)" requirements.txt > requirements-nonsci.txt

# Install remaining requirements
RUN pip install --prefer-binary -r requirements-nonsci.txt

# Expose port (Flask default)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
