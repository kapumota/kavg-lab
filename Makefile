.PHONY: validate clean

validate:
	bash scripts/validate.sh

clean:
	rm -rf target
	rm -rf evidence
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.csv" -delete
	find sample_outputs -type f -name "*.json" -delete
