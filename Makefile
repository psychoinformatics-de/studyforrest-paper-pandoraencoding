PYTHON = python3

#CLONEURL = http://psydata.ovgu.de/forrest_gump/.git
CLONEURL = medusa.ovgu.de:/home/data/psyinf/forrest_gump/anondata

all: data folders figures paper.pdf

data:
	[ ! -d data ] && git clone $(CLONEURL) data || git -C data pull
# example for how to obtain data from the annex
##	cd data && git annex get \
#		sub*/BOLD/task006_run*/bold_moco*.txt \
#		sub*/BOLD/task006_run*/bold_moco*_tsnr.nii.gz

paper.pdf: figures
	$(MAKE) -C paper all
	cp paper/p.pdf paper.pdf

figures: data folders pymvpa
	$(PYTHON) code/pandora_preprocessing.py 
	$(PYTHON) code/pandora_encoding.py
	$(PYTHON) code/validate_encoding.py
	$(PYTHON) code/make_plots.py

folders:
	mkdir preprocessed
	mkdir preprocessed/3T
	mkdir preprocessed/7T
	mkdir encoding
	mkdir encoding/3T
	mkdir encoding/7T
	mkdir validation
	mkdir validation/3T
	mkdir validation/7T

pymvpa:
	git clone https://github.com/PyMVPA/PyMVPA.git pymvpa

clean:
	-rm -f paper.pdf
	-rm -rf paper/pics/generated/*.svg
	$(MAKE) -C paper clean

distclean: clean
	-chmod -R u+w data/.git/annex/objects/*
	-rm -rf data pymvpa

.PHONY: figures clean distclean
