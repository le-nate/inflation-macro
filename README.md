# inflation-macro
Use `conda env create -f environment.yml` to create virtual environment
Use `pip install -e .` in cmd to install package(s).
Run `conda activate inflation` to run virtual environment

## API access
You will need to create personal accounts on the different database's websites and request API credentials. Save them to a file named `.env` in a root folder.

See `.example_env` for how you should store the credentials.

<b>Make sure that the file name is exactly `.env` (nothing before the `.` nor after `env`) so that your credentials are not exposed in your git repository!</b>

## References
Project structure inspired by [The Good Research Code Handbook](https://goodresearch.dev/#the-good-research-code-handbook) by Patrick Mineault.