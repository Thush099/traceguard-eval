VENV_PY := $(shell [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3)
PYTHONPATH := src

.PHONY: test benchmark fuzz export serve docker-build docker-run clean

test:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PY) -m pytest -v

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PY) main.py --benchmark

fuzz:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PY) main.py --fuzz --runs 30 --seed 42

export:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PY) main.py --benchmark --export-html reports/traceguard-audit.html

serve:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PY) main.py --serve --port 8000

docker-build:
	docker build -t traceguard-eval:latest .

docker-run:
	docker run --rm -p 8000:8000 traceguard-eval:latest

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__ reports/ *.html
