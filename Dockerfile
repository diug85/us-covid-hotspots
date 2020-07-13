# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./covid_tracker.py /app
COPY ./requirements.txt /app


# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
#EXPOSE 80

CMD [ "python", "./covid_tracker.py" ]
