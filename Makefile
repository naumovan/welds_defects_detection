run:
	streamlit run bin.py

install:
	pip install -e .

get_lfs:
	git lfs install
	git lfs pull
