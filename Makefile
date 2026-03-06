.PHONY: fmt precommit precommit-install

fmt:
	pre-commit run runic --all-files

precommit:
	pre-commit run --all-files

precommit-install:
	pre-commit install
