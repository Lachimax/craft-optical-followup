FROM python:3.12
WORKDIR /

# Install the package
ADD https://github.com/Lachimax/craft-optical-followup.git /craft-optical-followup/
WORKDIR /craft-optical-followup/
RUN pip install . -r requirements.txt

# CMD ls -a craft-optical-followup/*