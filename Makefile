SHELL := /bin/zsh
.PHONY: venv

install:
	conda env create --prefix venv --file environment.yml

clean:
	conda remove --prefix venv --all -y

predict:
	conda run --prefix venv python src/main.py


