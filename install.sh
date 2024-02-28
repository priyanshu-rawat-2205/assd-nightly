# create venv
python3 -m venv $1

# activate venv
source $1/bin/activate

# upgrade pip
python3 -m pip install --upgrade pip

# install requirements (dependencies)
pip install -r requirements.txt