# Use
## Docker
### Build Image
`docker build -t anomaly-dev-app:latest .`

### Run Container
`docker run -it -p 8080:8080`
Flask app is listening on port 8080 so we bind the container to port 8080.

### Run Container and Attach to Bash for Live Development
`docker run -it -p 8080:8080 --mount src="$(pwd)",target=/home,type=bind anomaly-dev-app:latest`

This mounts the current directory to the container at `/home/`. To do live development `cd` into `/home/app/` and run `python main.py`. Then open `main.py` in your code editor of choice and develop away. The Flask development server will restart every time you save changes to `main.py`.


## Local Installation
Install [tensorflow](https://www.tensorflow.org/install) and the packages listed in `requirements.txt` to a python virtual environment or conda environment and run Flask app locally.