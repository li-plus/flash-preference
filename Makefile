lint:
	autoflake -i --remove-all-unused-imports **/*.py
	isort .
	black . --line-length 120

test:
	pytest -sv tests/test_flash_pref.py
