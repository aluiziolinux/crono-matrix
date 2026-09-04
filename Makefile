PYTHON ?= .venv/bin/python

.PHONY: setup llama browser desktop web test release-check

setup:
	./scripts/setup.sh

llama:
	./scripts/bootstrap_llama_cpp.sh

browser:
	./scripts/bootstrap_llama_cpp.sh --with-browser

desktop:
	$(PYTHON) launch_model_gui.py

web:
	$(PYTHON) launch_model_web.py

test:
	$(PYTHON) -m compileall -q launch_model_core.py launch_model_ctk.py launch_model_gui.py launch_model_web.py web tests scripts
	$(PYTHON) -m unittest discover -s tests -v

release-check:
	$(PYTHON) scripts/release_check.py
