check:
	flake8 --max-line-length=120 ./Autoformer
	pycodestyle --max-line-length=120 ./Autoformer
	pylint --disable=all --enable=W0221 ./Autoformer