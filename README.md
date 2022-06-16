# Data analysis
- Project ToVoice
- This repo is the product of a 10 day project which was carried out at the end of a Data Science bootcamp at Le Wagon
- Data Source: https://github.com/auspicious3000/autovc
- Type of analysis: Zero shot voice conversion - (result : the model still needs to be trained on voices to work)

We looked at the original model's thesis and github repo to be able to work on this reconstruction.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for tovoice in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/tovoice`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "tovoice"
git remote add origin git@github.com:{group}/tovoice.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
tovoice-run
```

# Install

Go to `https://github.com/{group}/tovoice` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/tovoice.git
cd tovoice
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
tovoice-run
```
