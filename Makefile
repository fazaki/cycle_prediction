build: cleanup pywheel
pystyle: style
docs: docstring
publish: publish_docs  # publish_examples
test: tox  # tox
upload: pypi
install: pip_install
all: style tox cleanup pywheel pypi

# publish requires to run below first. only user of mapcore and publish
# ssh-copy-id -p 28222 mapcore@superai.jp.ao.ericsson.se

PROJECTNAME   = cycle_prediction

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# GROUPNAME     = 

PUBLISH_HOST   ?= superai.jp.ao.ericsson.se
PUBLISH_PORT   ?= 28222
PUBLISH_USER   ?= mapcore

style:
	@echo "### projectname $(PROJECTNAME) ..."
	@echo "### running pycodestyle ..."
	@pycodestyle $(PROJECTNAME)/
	@pycodestyle tests/
	@echo "### running pyflakes ..."
	@pyflakes $(PROJECTNAME)/
	@pyflakes tests/
	@echo "### finished style ..."

tox:
	@echo "### running tox ..."
	@tox

cleanup:
	@echo "### rm -rf ./build ./dist"
	@rm -rf ./build ./dist

pywheel:
	@echo "### packaging python ..."
	@python setup.py bdist_wheel

docstring:
	@echo "### clean build ..."
	@cd docs && $(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) && cd ..
	@echo "### build docstring html ..."
	@cd docs && $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) && cd ..


# publish_docs:
# 	@echo "### publish docs..."
# 	@echo "### make $(PROJECTNAME) in $(PUBLISH_USER).."
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "cd /home/$(PUBLISH_USER)/public_html && mkdir -p $(PROJECTNAME)"
# 	@echo "### publish docs to $(PUBLISH_USER)..."
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "cd /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME) && rm -rf html"
# 	@scp -P $(PUBLISH_PORT) -rp docs/build/html/ $(PUBLISH_USER)@$(PUBLISH_HOST):/home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "chmod -R 755 /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME) && \
# 		rm -rf /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/docs && \
# 		mv /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/html /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/docs"
# 	@echo "https://$(PUBLISH_HOST):$(PUBLISH_PORT)/~$(PUBLISH_USER)/$(PROJECTNAME)/docs/"
# 	@echo "### fin publish..."

# publish_examples:
# 	@echo "### publish examples..."
# 	@echo "### make $(PROJECTNAME) in $(PUBLISH_USER).."
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "cd /home/$(PUBLISH_USER)/public_html && mkdir -p $(PROJECTNAME)"
# 	@echo "### publish docs to $(PUBLISH_USER)..."
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "cd /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME) && rm -rf examples"
# 	@scp -P $(PUBLISH_PORT) -rp examples/ $(PUBLISH_USER)@$(PUBLISH_HOST):/home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)
# 	@ssh -p $(PUBLISH_PORT) $(PUBLISH_USER)@$(PUBLISH_HOST) "chmod -R 755 /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME) && \
# 		rm -rf /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/notebooks && \
# 		mv /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/examples /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/notebooks && \
# 		echo "Options +Indexes" > /home/$(PUBLISH_USER)/public_html/$(PROJECTNAME)/notebooks/.htaccess"
# 	@echo "https://$(PUBLISH_HOST):$(PUBLISH_PORT)/~$(PUBLISH_USER)/$(PROJECTNAME)/notebooks/"
# 	@echo "### fin publish..."

pypi:
	@echo "### move whl to testpypi ..."
	# @twine upload --repository testpypi dist/*
	@twine upload dist/*
	@echo "### DONE !"

pip_install:
	@echo "### pip install..."
	@pip install -i https://test.pypi.org/simple/ $(PROJECTNAME)
